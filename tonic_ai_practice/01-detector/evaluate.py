"""Evaluation harness for detector.py.

Step 0: load the sample, put gold and predictions into one span shape,
and prove the character offsets are trustworthy before measuring anything.
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

from detector import detect

DATA_PATH = Path(__file__).parent / "data" / "train_en_1500.jsonl"


def load_docs(path=DATA_PATH, limit=None):
    """Read the cached JSONL into a list of raw dicts."""
    docs = []
    with open(path) as f:
        for line in f:
            docs.append(json.loads(line))
            if limit is not None and len(docs) >= limit:
                break
    return docs


# --- Step 1: label reconciliation ------------------------------------------
#
# Gold uses 56 fine-grained types; the detector emits 7 names that don't match
# any of them. Both sides are mapped into one shared eval space. Every span
# keeps its original `label` so severity can be analysed on gold's own terms.

OOV = "OUT_OF_VOCAB"  # gold PII the detector has no pattern for at all

PRED_TO_EVAL = {
    "EMAIL": "EMAIL",
    "PHONE": "PHONENUMBER",
    "SSN": "SSN",
    "DATE": "DATE",
    "CREDITCARD": "CREDITCARDNUMBER",
    "ZIP": "ZIPCODE",
    "PERSON": "PERSON",
}

GOLD_TO_EVAL = {
    "EMAIL": "EMAIL",
    "PHONENUMBER": "PHONENUMBER",
    "SSN": "SSN",
    "CREDITCARDNUMBER": "CREDITCARDNUMBER",
    "ZIPCODE": "ZIPCODE",
    # DATE and DOB are format-identical in this corpus, so one eval label.
    # The original label survives for the severity split.
    "DATE": "DATE",
    "DOB": "DATE",
    # A person is three separate gold spans. PREFIX ("Dr.") is deliberately
    # excluded: it identifies nobody on its own.
    "FIRSTNAME": "PERSON",
    "MIDDLENAME": "PERSON",
    "LASTNAME": "PERSON",
}


# --- Step 2: severity tiering ----------------------------------------------
#
# Decided before any per-type result is computed, so it can't be bent to fit
# the numbers. The corpus is going to a model-training vendor: a missed entity
# is trained into weights and cannot be retracted, a spurious one costs data
# quality. So the tiers rank by "does this identify one person, or grant
# access", not by how sensitive the word sounds.

HIGH, MEDIUM, LOW = "HIGH", "MEDIUM", "LOW"

SEVERITY = {
    # credentials and secrets — direct access, re-emittable by a trained model
    "PASSWORD": HIGH, "PIN": HIGH, "CREDITCARDCVV": HIGH,
    # government and financial identifiers
    "SSN": HIGH, "CREDITCARDNUMBER": HIGH, "IBAN": HIGH, "ACCOUNTNUMBER": HIGH,
    "BITCOINADDRESS": HIGH, "ETHEREUMADDRESS": HIGH, "LITECOINADDRESS": HIGH,
    # direct contact and identity
    "EMAIL": HIGH, "PHONENUMBER": HIGH, "USERNAME": HIGH, "DOB": HIGH,
    # A bare given name identifies almost nobody, so it is not HIGH on its own.
    # severity() takes the max over a merged span's parts, so a full name still
    # rates HIGH through LASTNAME while a lone "Bud" stays MEDIUM.
    "FIRSTNAME": MEDIUM, "MIDDLENAME": MEDIUM, "LASTNAME": HIGH,
    # home address components
    "STREET": HIGH, "BUILDINGNUMBER": HIGH, "SECONDARYADDRESS": HIGH,
    "NEARBYGPSCOORDINATE": HIGH,
    # device and network identifiers (Breyer: a dynamic IP is personal data)
    "IP": HIGH, "IPV4": HIGH, "IPV6": HIGH, "MAC": HIGH, "PHONEIMEI": HIGH,
    "VEHICLEVIN": HIGH, "VEHICLEVRM": HIGH,

    # Quasi-identifiers — dangerous in combination, not alone. The honest caveat:
    # this table tiers entities IN ISOLATION while the real risk is COMBINATORIAL.
    # ZIP + DOB + SEX re-identifies ~87% of US adults (Sweeney) and all three
    # occur in this corpus, yet each is tiered separately here. A per-entity
    # table cannot express that; it needs a document-level rule (Step 6).
    "ZIPCODE": MEDIUM,
    "CITY": MEDIUM, "COUNTY": MEDIUM, "STATE": MEDIUM,
    "AGE": MEDIUM, "DATE": MEDIUM, "TIME": MEDIUM,
    "COMPANYNAME": MEDIUM, "JOBTITLE": MEDIUM, "URL": MEDIUM,
    "SEX": MEDIUM, "GENDER": MEDIUM,     # GDPR Art.9 sensitive, but not identifying alone
    "USERAGENT": MEDIUM, "MASKEDNUMBER": MEDIUM, "BIC": MEDIUM,

    # context — does not meaningfully narrow who this is
    "ACCOUNTNAME": LOW,   # values are "Home Loan Account", not a person
    "AMOUNT": LOW, "CREDITCARDISSUER": LOW, "PREFIX": LOW,
    "CURRENCY": LOW, "CURRENCYCODE": LOW, "CURRENCYNAME": LOW,
    "CURRENCYSYMBOL": LOW, "EYECOLOR": LOW, "HEIGHT": LOW,
    "JOBAREA": LOW, "JOBTYPE": LOW, "ORDINALDIRECTION": LOW,
}

_TIER_ORDER = {LOW: 0, MEDIUM: 1, HIGH: 2}


def severity(span):
    """Tier of a gold span. A merged PERSON takes the highest tier of its parts."""
    parts = span.get("parts", [span["label"]])
    return max((SEVERITY[p] for p in parts), key=_TIER_ORDER.__getitem__)


NAME_PARTS = {"FIRSTNAME", "MIDDLENAME", "LASTNAME"}


def merge_name_parts(spans, source):
    """Fuse whitespace-adjacent name parts into one PERSON span.

    Gold splits a person into FIRSTNAME/MIDDLENAME/LASTNAME; the detector
    emits one span per name. Without this, one prediction gets scored
    against three gold spans and two of them count as missed.

    Only merges across a gap that is empty or whitespace, so "Alice and Bob"
    stays two people. PREFIX is not a name part and never merges.
    """
    spans = sorted(spans, key=lambda s: (s["start"], s["end"]))
    out, run = [], []

    def flush():
        if not run:
            return
        if len(run) == 1:
            out.append(run[0])
        else:
            out.append({
                "doc_id": run[0]["doc_id"],
                "start": run[0]["start"],
                "end": run[-1]["end"],
                "label": "PERSON",
                "eval_label": "PERSON",
                "parts": [r["label"] for r in run],
                "text": source[run[0]["start"]:run[-1]["end"]],
            })
        run.clear()

    for s in spans:
        if s["label"] not in NAME_PARTS:
            flush()
            out.append(s)
        elif run and source[run[-1]["end"]:s["start"]].strip() == "":
            run.append(s)
        else:
            flush()
            run.append(s)
    flush()
    return out


def gold_spans(doc, merge=True):
    """privacy_mask -> canonical spans. `text` comes from gold's own value field."""
    spans = [
        {
            "doc_id": str(doc["id"]),
            "start": s["start"],
            "end": s["end"],
            "label": s["label"],
            "eval_label": GOLD_TO_EVAL.get(s["label"], OOV),
            "text": s["value"],
        }
        for s in doc["privacy_mask"]
    ]
    return merge_name_parts(spans, doc["source_text"]) if merge else spans


def pred_spans(doc):
    """detect() -> canonical spans. `text` is sliced from the unmodified source."""
    source = doc["source_text"]
    return [
        {
            "doc_id": str(doc["id"]),
            "start": s["start"],
            "end": s["end"],
            "label": s["label"],
            "eval_label": PRED_TO_EVAL[s["label"]],
            "text": source[s["start"]:s["end"]],
        }
        for s in detect(source)
    ]


def check_offsets(docs):
    """Gold offsets must slice back to gold values. Returns the failures.

    Runs on UNMERGED gold: merged spans take their text from the source, so
    checking them would compare a slice against itself and always pass.
    """
    bad = []
    for doc in docs:
        source = doc["source_text"]
        for s in gold_spans(doc, merge=False):
            if source[s["start"]:s["end"]] != s["text"]:
                bad.append((s, source[s["start"]:s["end"]]))
    return bad


# --- Step 3: alignment -----------------------------------------------------

CORRECT, BOUNDARY, TYPE_CONF = "correct", "boundary", "type_confusion"
MISSED, SPURIOUS = "missed", "spurious"


def overlap(a, b):
    """Characters two spans share. Adjacent spans (a.end == b.start) share 0."""
    return max(0, min(a["end"], b["end"]) - max(a["start"], b["start"]))


def align(gold, pred):
    """Greedy one-to-one alignment of predictions onto gold spans.

    Candidates rank by (exact, same_label, overlap), so a prediction that
    straddles two gold spans claims the one it AGREES with rather than the
    one it merely covers most. One gold span takes at most one prediction
    and vice versa.
    """
    cands = []
    for gi, g in enumerate(gold):
        for pi, p in enumerate(pred):
            ov = overlap(g, p)
            if ov == 0:
                continue
            same = g["eval_label"] == p["eval_label"]
            exact = same and g["start"] == p["start"] and g["end"] == p["end"]
            cands.append((exact, same, ov, gi, pi))
    cands.sort(reverse=True)

    taken_g, taken_p, pairs = set(), set(), []
    for exact, same, _ov, gi, pi in cands:
        if gi in taken_g or pi in taken_p:
            continue
        taken_g.add(gi)
        taken_p.add(pi)
        pairs.append((gi, pi, exact, same))
    return pairs


def bucket_doc(doc):
    """Put every gold span and every prediction in exactly one bucket."""
    gold = gold_spans(doc, merge=True)
    pred = pred_spans(doc)
    pairs = align(gold, pred)

    g_bucket = [MISSED] * len(gold)
    p_bucket = [SPURIOUS] * len(pred)

    for gi, pi, exact, same in pairs:
        b = CORRECT if exact else (BOUNDARY if same else TYPE_CONF)
        g_bucket[gi] = b
        p_bucket[pi] = b

    # A prediction left unmatched that still touches gold is REDUNDANT, not
    # spurious — for redaction, covering the same entity twice is harmless.
    # It stays in the spurious bucket per the five-bucket convention, but is
    # counted separately so over-redaction isn't overstated.
    matched_p = {pi for _gi, pi, _e, _s in pairs}
    redundant = sum(
        1 for pi, p in enumerate(pred)
        if pi not in matched_p and any(overlap(g, p) for g in gold)
    )
    return gold, pred, g_bucket, p_bucket, redundant, pairs


def _span(start, end, label):
    return {"doc_id": "t", "start": start, "end": end,
            "label": label, "eval_label": label, "text": ""}


def selftest():
    """The edge cases an interviewer probes. Cheap, and they'd otherwise bite."""
    # adjacent spans share no characters and must not be fused
    assert overlap(_span(0, 3, "PERSON"), _span(3, 10, "EMAIL")) == 0

    # a prediction straddling two gold spans claims the one it AGREES with,
    # not the one it covers most
    gold = [_span(0, 5, "ZIPCODE"), _span(4, 12, "PERSON")]
    pairs = align(gold, [_span(2, 12, "PERSON")])
    assert len(pairs) == 1 and pairs[0][0] == 1, pairs

    # one gold, two predictions: the exact one wins, the other is left over
    pairs = align([_span(0, 10, "EMAIL")],
                  [_span(0, 8, "EMAIL"), _span(0, 10, "EMAIL")])
    assert len(pairs) == 1 and pairs[0][1] == 1 and pairs[0][2] is True, pairs

    # duplicate identical predictions claim the gold span once
    assert len(align([_span(0, 5, "SSN")],
                     [_span(0, 5, "SSN"), _span(0, 5, "SSN")])) == 1

    # a prediction touching no gold matches nothing
    assert align([_span(0, 5, "SSN")], [_span(20, 25, "PERSON")]) == []

    # zero-length spans can never match
    assert overlap(_span(5, 5, "SSN"), _span(0, 10, "SSN")) == 0
    print("selftest OK")


def alignment_report(docs):
    """Bucket totals, plus the invariant that nothing was lost or double-counted."""
    g_counts, p_counts = Counter(), Counter()
    n_gold = n_pred = redundant = 0

    for doc in docs:
        gold, pred, gb, pb, red, _pairs = bucket_doc(doc)
        g_counts.update(gb)
        p_counts.update(pb)
        n_gold += len(gold)
        n_pred += len(pred)
        redundant += red

    print("\ngold side:")
    for b in (CORRECT, BOUNDARY, TYPE_CONF, MISSED):
        print(f"  {b:16s} {g_counts[b]:5d}")
    print(f"  {'TOTAL':16s} {sum(g_counts.values()):5d}  (gold spans: {n_gold})")

    print("\nprediction side:")
    for b in (CORRECT, BOUNDARY, TYPE_CONF, SPURIOUS):
        print(f"  {b:16s} {p_counts[b]:5d}")
    print(f"  {'TOTAL':16s} {sum(p_counts.values()):5d}  (predictions: {n_pred})")
    print(f"\n  of which redundant (touch gold, but it was already claimed): "
          f"{redundant}")
    print(f"  truly spurious (touch no gold at all): "
          f"{p_counts[SPURIOUS] - redundant}")

    assert sum(g_counts.values()) == n_gold, "a gold span fell out of the buckets"
    assert sum(p_counts.values()) == n_pred, "a prediction fell out of the buckets"
    print("\n  invariant OK: every span in exactly one bucket")


# --- Step 4: error bucketing per type and per severity ---------------------


def bucket_tables(docs):
    """Bucket counts keyed by gold label, by prediction label, and by tier."""
    gold_rows = defaultdict(Counter)
    pred_rows = defaultdict(Counter)
    tier_rows = defaultdict(Counter)
    confusion = Counter()   # (predicted eval label, true gold label)

    for doc in docs:
        gold, pred, gb, pb, _red, pairs = bucket_doc(doc)
        g2p = {gi: pi for gi, pi, _e, _s in pairs}
        for gi, g in enumerate(gold):
            gold_rows[g["eval_label"]][gb[gi]] += 1
            tier_rows[severity(g)][gb[gi]] += 1
            if gb[gi] == TYPE_CONF:
                confusion[(pred[g2p[gi]]["eval_label"], g["label"])] += 1
        for pi, p in enumerate(pred):
            pred_rows[p["eval_label"]][pb[pi]] += 1

    return gold_rows, pred_rows, tier_rows, confusion


def _rates(c):
    """exact, relaxed, covered. `covered` ignores the label: was it redacted?"""
    total = sum(c.values())
    if not total:
        return 0.0, 0.0, 0.0
    return (c[CORRECT] / total,
            (c[CORRECT] + c[BOUNDARY]) / total,
            (c[CORRECT] + c[BOUNDARY] + c[TYPE_CONF]) / total)


def bucket_report(docs):
    gold_rows, pred_rows, tier_rows, confusion = bucket_tables(docs)

    print("\nRECALL by gold label   (covered = touched by any prediction)")
    print(f"{'label':16s} {'gold':>5s} {'corr':>5s} {'bound':>6s} "
          f"{'conf':>5s} {'miss':>5s} {'exact':>7s} {'relax':>7s} {'cover':>7s}")
    for lab in sorted(gold_rows, key=lambda x: -sum(gold_rows[x].values())):
        c = gold_rows[lab]
        n = sum(c.values())
        e, r, v = _rates(c)
        print(f"{lab:16s} {n:5d} {c[CORRECT]:5d} {c[BOUNDARY]:6d} "
              f"{c[TYPE_CONF]:5d} {c[MISSED]:5d} "
              f"{e:6.1%} {r:6.1%} {v:6.1%}")

    print("\nPRECISION by predicted label")
    print(f"{'label':16s} {'pred':>5s} {'corr':>5s} {'bound':>6s} "
          f"{'conf':>5s} {'spur':>5s} {'exact':>7s} {'relax':>7s}")
    for lab in sorted(pred_rows, key=lambda x: -sum(pred_rows[x].values())):
        c = pred_rows[lab]
        n = sum(c.values())
        e, r, _v = _rates(c)
        print(f"{lab:16s} {n:5d} {c[CORRECT]:5d} {c[BOUNDARY]:6d} "
              f"{c[TYPE_CONF]:5d} {c[SPURIOUS]:5d} {e:6.1%} {r:6.1%}")

    print("\nRECALL by severity tier")
    print(f"{'tier':10s} {'gold':>5s} {'exact':>8s} {'relax':>8s} {'cover':>8s}")
    for tier in (HIGH, MEDIUM, LOW):
        c = tier_rows[tier]
        e, r, v = _rates(c)
        print(f"{tier:10s} {sum(c.values()):5d} {e:7.1%} {r:7.1%} {v:7.1%}")

    print("\nTYPE CONFUSION: predicted label -> what it actually was (top 15)")
    for (pl, gl), n in confusion.most_common(15):
        print(f"  {pl:16s} -> {gl:20s} {n:4d}")


def reconciliation_report(docs, merge=True):
    """How much of the gold PII is even addressable by this detector?"""
    in_vocab = Counter()   # eval_label -> gold spans the detector could match
    oov = Counter()        # original gold label -> spans with no pattern at all
    docs_all_oov = 0

    for doc in docs:
        gold = gold_spans(doc, merge=merge)
        for s in gold:
            if s["eval_label"] == OOV:
                oov[s["label"]] += 1
            else:
                in_vocab[s["eval_label"]] += 1
        if gold and all(s["eval_label"] == OOV for s in gold):
            docs_all_oov += 1

    n_in, n_oov = sum(in_vocab.values()), sum(oov.values())
    total = n_in + n_oov

    print(f"\ngold spans in detector vocabulary : {n_in:5d}  ({n_in / total:.1%})")
    print(f"gold spans OUT of vocabulary      : {n_oov:5d}  ({n_oov / total:.1%})")
    print(f"docs where every gold span is OOV : {docs_all_oov} of {len(docs)}")

    print(f"\nin-vocabulary gold, by eval label ({len(in_vocab)} types):")
    for lab, c in in_vocab.most_common():
        print(f"  {lab:18s} {c:5d}")

    print(f"\nout-of-vocabulary gold ({len(oov)} types, top 20):")
    for lab, c in oov.most_common(20):
        print(f"  {lab:20s} {c:5d}")


def merge_report(docs):
    """How many name parts got fused, and into runs of what length."""
    runs = Counter()
    examples = []
    for doc in docs:
        for s in gold_spans(doc, merge=True):
            if "parts" in s:
                runs[len(s["parts"])] += 1
                if len(examples) < 4:
                    examples.append((s["text"], s["parts"]))
    fused = sum(n * c for n, c in runs.items())
    print(f"\nname-part merges: {sum(runs.values())} PERSON spans built "
          f"from {fused} gold parts")
    for n in sorted(runs):
        print(f"  runs of {n}: {runs[n]}")
    for t, p in examples:
        print(f"  e.g. {t!r} <- {p}")


def severity_report(docs):
    """Gold PII by tier, and how much of each tier has no pattern at all."""
    untiered = {s["label"] for d in docs
                for s in gold_spans(d, merge=False) if s["label"] not in SEVERITY}
    if untiered:
        raise KeyError(f"gold labels with no severity tier: {sorted(untiered)}")

    by_tier, oov_by_tier = Counter(), Counter()
    high_oov_types = Counter()
    docs_with_high = docs_high_oov = 0

    for doc in docs:
        gold = gold_spans(doc, merge=True)
        high = [s for s in gold if severity(s) == HIGH]
        docs_with_high += bool(high)
        docs_high_oov += any(s["eval_label"] == OOV for s in high)
        for s in gold:
            tier = severity(s)
            by_tier[tier] += 1
            if s["eval_label"] == OOV:
                oov_by_tier[tier] += 1
                if tier == HIGH:
                    high_oov_types.update(s.get("parts", [s["label"]]))

    print(f"\n{'tier':8s} {'gold':>7s} {'no pattern':>12s} {'addressable':>13s}")
    for tier in (HIGH, MEDIUM, LOW):
        n, o = by_tier[tier], oov_by_tier[tier]
        print(f"{tier:8s} {n:7d} {o:7d} ({o / n:3.0%}) {n - o:13d}")

    print(f"\ndocs with >=1 HIGH entity          : {docs_with_high} of {len(docs)}")
    print(f"docs where a HIGH entity has NO pattern: {docs_high_oov} "
          f"({docs_high_oov / len(docs):.0%})  <- leaks regardless of matching")

    print("\nHIGH-severity types with no pattern in detector.py:")
    for lab, c in high_oov_types.most_common():
        print(f"  {lab:20s} {c:5d}")


if __name__ == "__main__":
    docs = load_docs()

    bad = check_offsets(docs)
    n_raw = sum(len(gold_spans(d, merge=False)) for d in docs)
    n_merged = sum(len(gold_spans(d, merge=True)) for d in docs)
    n_pred = sum(len(pred_spans(d)) for d in docs)

    print(f"docs={len(docs)}  pred_spans={n_pred}")
    print(f"gold spans: {n_raw} raw -> {n_merged} after name merge")
    print(f"offset mismatches: {len(bad)}")
    for s, sliced in bad[:5]:
        print(f"  doc={s['doc_id']} {s['label']} value={s['text']!r} slice={sliced!r}")

    merge_report(docs)

    print("\n" + "=" * 60)
    print("BEFORE merge")
    print("=" * 60)
    reconciliation_report(docs, merge=False)

    print("\n" + "=" * 60)
    print("AFTER merge  <- the denominator used from here on")
    print("=" * 60)
    reconciliation_report(docs, merge=True)

    print("\n" + "=" * 60)
    print("SEVERITY")
    print("=" * 60)
    severity_report(docs)

    print("\n" + "=" * 60)
    print("ALIGNMENT")
    print("=" * 60)
    selftest()
    alignment_report(docs)

    print("\n" + "=" * 60)
    print("BUCKETS BY TYPE AND SEVERITY")
    print("=" * 60)
    bucket_report(docs)
