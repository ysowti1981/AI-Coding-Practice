"""Evaluation harness for problem 03.

Everything here works from the two JSONL files only. Nothing is justified by
reading pipeline.py — see the contamination note in PROBLEM.md.
"""

import json
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
SOURCE_PATH = DATA_DIR / "train_en_2000.jsonl"
REDACTED_PATH = DATA_DIR / "redacted.jsonl"


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def load_joined():
    """Join the redacted output onto the source corpus by document id."""
    src = read_jsonl(SOURCE_PATH)
    by_id = {str(r["id"]): r for r in src}
    assert len(by_id) == len(src), "duplicate id in the source corpus"

    red = read_jsonl(REDACTED_PATH)
    red_ids = [r["doc_id"] for r in red]
    assert len(set(red_ids)) == len(red), "duplicate doc_id in the redacted file"
    assert set(red_ids) == set(by_id), (
        f"redacted file does not cover the corpus: "
        f"{len(set(by_id) - set(red_ids))} missing, "
        f"{len(set(red_ids) - set(by_id))} extra"
    )

    docs = []
    for r in red:
        s = by_id[r["doc_id"]]
        docs.append({
            "doc_id": r["doc_id"],
            "source_text": s["source_text"],
            "original_text": r["original_text"],
            "redacted_text": r["redacted_text"],
            "spans_replaced": r["spans_replaced"],
            "gold": s["privacy_mask"],
        })
    print(f"joined {len(docs)} documents, join is total")
    return docs


def verify_echo(docs):
    """`original_text` must equal `source_text`.

    If the pipeline did not echo its input faithfully, every comparison
    downstream is against the wrong baseline.
    """
    bad = [d for d in docs if d["original_text"] != d["source_text"]]
    print(f"original_text != source_text: {len(bad)} of {len(docs)}")
    for d in bad[:3]:
        print(f"  doc {d['doc_id']}")
        print(f"    source   {d['source_text'][:90]!r}")
        print(f"    original {d['original_text'][:90]!r}")
    return bad


def show_pairs(docs, n=20, width=260):
    """Original above, redacted below. The cheapest information in the hour."""
    for d in docs[:n]:
        gold = sorted(d["gold"], key=lambda s: s["start"])
        print(f"\n{'=' * 78}\ndoc {d['doc_id']}   "
              f"gold spans={len(gold)}   spans_replaced={d['spans_replaced']}")
        print(f"  ORIG: {d['original_text'][:width]}")
        print(f"  RED : {d['redacted_text'][:width]}")
        print(f"  gold: {[(s['label'], s['value']) for s in gold][:8]}")


# Severity tiers, carried over unchanged from 01-detector so the two problems
# stay comparable. One call worth noting: the recipient here is a research
# partner, where re-identification by combination dominates, so CITY/ZIPCODE
# arguably belong higher than they sit. Left as-is deliberately.
HIGH, MEDIUM, LOW = "HIGH", "MEDIUM", "LOW"

SEVERITY = {
    "PASSWORD": HIGH, "PIN": HIGH, "CREDITCARDCVV": HIGH,
    "SSN": HIGH, "CREDITCARDNUMBER": HIGH, "IBAN": HIGH, "ACCOUNTNUMBER": HIGH,
    "BITCOINADDRESS": HIGH, "ETHEREUMADDRESS": HIGH, "LITECOINADDRESS": HIGH,
    "EMAIL": HIGH, "PHONENUMBER": HIGH, "USERNAME": HIGH, "DOB": HIGH,
    "FIRSTNAME": MEDIUM, "MIDDLENAME": MEDIUM, "LASTNAME": HIGH,
    "STREET": HIGH, "BUILDINGNUMBER": HIGH, "SECONDARYADDRESS": HIGH,
    "NEARBYGPSCOORDINATE": HIGH,
    "IP": HIGH, "IPV4": HIGH, "IPV6": HIGH, "MAC": HIGH, "PHONEIMEI": HIGH,
    "VEHICLEVIN": HIGH, "VEHICLEVRM": HIGH,
    "ZIPCODE": MEDIUM, "CITY": MEDIUM, "COUNTY": MEDIUM, "STATE": MEDIUM,
    "AGE": MEDIUM, "DATE": MEDIUM, "TIME": MEDIUM,
    "COMPANYNAME": MEDIUM, "JOBTITLE": MEDIUM, "URL": MEDIUM,
    "SEX": MEDIUM, "GENDER": MEDIUM,
    "USERAGENT": MEDIUM, "MASKEDNUMBER": MEDIUM, "BIC": MEDIUM,
    "ACCOUNTNAME": LOW, "AMOUNT": LOW, "CREDITCARDISSUER": LOW, "PREFIX": LOW,
    "CURRENCY": LOW, "CURRENCYCODE": LOW, "CURRENCYNAME": LOW,
    "CURRENCYSYMBOL": LOW, "EYECOLOR": LOW, "HEIGHT": LOW,
    "JOBAREA": LOW, "JOBTYPE": LOW, "ORDINALDIRECTION": LOW,
}

MIN_LEN = 4  # headline excludes short values, where substring search is noisy


def residual_pii(docs):
    """Is each gold value still present, verbatim, in the released text?

    Offset-independent substring search. This is a LOWER bound on leakage:
    a value that survives in fragments does not count here.
    """
    from collections import Counter

    by_tier = Counter()
    leaked_by_tier = Counter()
    by_label = Counter()
    leaked_by_label = Counter()
    short_leaked = 0
    docs_leaking_high = 0
    total_gold = total_replaced = 0
    examples = []

    for d in docs:
        red = d["redacted_text"]
        doc_high = False
        total_gold += len(d["gold"])
        total_replaced += d["spans_replaced"]

        for span in d["gold"]:
            value, label = span["value"], span["label"]
            tier = SEVERITY.get(label, MEDIUM)
            present = value in red

            if len(value) < MIN_LEN:
                if present:
                    short_leaked += 1
                continue

            by_tier[tier] += 1
            by_label[label] += 1
            if present:
                leaked_by_tier[tier] += 1
                leaked_by_label[label] += 1
                if tier == HIGH:
                    doc_high = True
                    if len(examples) < 8:
                        examples.append((d["doc_id"], label, value))
        docs_leaking_high += doc_high

    print(f"\ngold spans {total_gold}   pipeline reports "
          f"{total_replaced} replaced "
          f"({total_replaced / total_gold:.1%} — the claimed 94%)")

    print(f"\nRESIDUAL PII — original value still present verbatim "
          f"(values >= {MIN_LEN} chars)")
    print(f"{'tier':8s} {'spans':>7s} {'leaked':>8s} {'rate':>8s}")
    for tier in (HIGH, MEDIUM, LOW):
        n, k = by_tier[tier], leaked_by_tier[tier]
        print(f"{tier:8s} {n:7d} {k:8d} {k / n:8.1%}" if n else
              f"{tier:8s} {n:7d} {k:8d} {'-':>8s}")
    n, k = sum(by_tier.values()), sum(leaked_by_tier.values())
    print(f"{'ALL':8s} {n:7d} {k:8d} {k / n:8.1%}")
    print(f"  (short values < {MIN_LEN} chars, excluded above: "
          f"{short_leaked} present — substring search is unreliable there)")

    print(f"\ndocuments retaining >=1 HIGH-severity value verbatim: "
          f"{docs_leaking_high} of {len(docs)} "
          f"({docs_leaking_high / len(docs):.1%})")

    print("\nworst labels by leak count")
    for label, k in leaked_by_label.most_common(12):
        print(f"  {label:20s} {k:4d} of {by_label[label]:4d} "
              f"({k / by_label[label]:.0%})  [{SEVERITY.get(label, '?')}]")

    print("\nexamples of HIGH-severity values still in the released text")
    for doc_id, label, value in examples:
        print(f"  doc {doc_id:>8s}  {label:16s} {value!r}")

    return docs_leaking_high


MIN_CONTEXT = 12  # ignore tiny between-span fragments; they match by chance


def replacement_integrity(docs):
    """Did replacements land where they were meant to?

    Three measurements, all from the output:

    1. Leak rate by a span's ORDINAL POSITION in its document. If each
       replacement knocks the following offsets out of alignment, the first
       span in a document should be safe and later ones progressively worse.

    2. Single-span documents vs multi-span. A document with one span has no
       preceding replacement, so its leak rate should equal the detector's
       drop rate. This is the natural experiment that isolates the effect.

    3. Collateral damage. The text BETWEEN gold spans is not PII and should
       survive untouched. Whether it does tests splicing without reference
       to any replacement pool.
    """
    from collections import Counter

    by_rank = Counter()
    leaked_by_rank = Counter()
    single_n = single_leaked = 0
    multi_n = multi_leaked = 0

    ctx_total = ctx_lost = 0
    docs_ctx_damaged = 0
    len_delta = []

    for d in docs:
        red, orig = d["redacted_text"], d["original_text"]
        gold = sorted(d["gold"], key=lambda s: s["start"])
        len_delta.append(len(red) - len(orig))

        for rank, span in enumerate(gold):
            if len(span["value"]) < MIN_LEN:
                continue
            present = span["value"] in red
            key = rank if rank < 4 else "4+"
            by_rank[key] += 1
            if present:
                leaked_by_rank[key] += 1
            if len(gold) == 1:
                single_n += 1
                single_leaked += present
            else:
                multi_n += 1
                multi_leaked += present

        # non-PII context: prefix, gaps between spans, suffix
        cuts = [0] + [c for s in gold for c in (s["start"], s["end"])] + [len(orig)]
        segments = [orig[cuts[i]:cuts[i + 1]] for i in range(0, len(cuts) - 1, 2)]
        damaged = False
        for seg in segments:
            if len(seg) < MIN_CONTEXT:
                continue
            ctx_total += 1
            if seg not in red:
                ctx_lost += 1
                damaged = True
        docs_ctx_damaged += damaged

    print("\nLEAK RATE BY SPAN POSITION IN ITS DOCUMENT")
    print(f"{'position':>10s} {'spans':>7s} {'leaked':>8s} {'rate':>8s}")
    for key in [0, 1, 2, 3, "4+"]:
        n, k = by_rank[key], leaked_by_rank[key]
        label = f"#{key}" if key != "4+" else "#4+"
        print(f"{label:>10s} {n:7d} {k:8d} {k / n:8.1%}" if n else "")

    print("\nTHE NATURAL EXPERIMENT — documents with one span vs many")
    if single_n == 0:
        print("  no single-span documents in this corpus (minimum is 2), so "
              "that comparison is unavailable.")
        print("  Position #0 above serves the same purpose: a document's first "
              "span has no preceding replacement, and it is the control.")
    else:
        print(f"  single-span docs   {single_n:5d} spans   "
              f"{single_leaked:4d} leaked   {single_leaked / single_n:6.1%}")
    print(f"  multi-span docs    {multi_n:5d} spans   "
          f"{multi_leaked:4d} leaked   {multi_leaked / multi_n:6.1%}")

    print("\nCOLLATERAL DAMAGE — non-PII text between spans")
    print(f"  context segments (>= {MIN_CONTEXT} chars)  {ctx_total:6d}")
    print(f"  no longer present verbatim          {ctx_lost:6d} "
          f"({ctx_lost / ctx_total:.1%})")
    print(f"  documents with damaged context      {docs_ctx_damaged:6d} of "
          f"{len(docs)} ({docs_ctx_damaged / len(docs):.1%})")

    len_delta.sort()
    mid = len(len_delta) // 2
    print("\nLENGTH CHANGE (redacted - original), chars")
    print(f"  min {len_delta[0]}   p25 {len_delta[len(len_delta) // 4]}   "
          f"median {len_delta[mid]}   p75 {len_delta[3 * len(len_delta) // 4]}   "
          f"max {len_delta[-1]}")




def minimum_fix_simulation(docs, seed=13, coverage=0.94):
    """Apply the same replacements RIGHT TO LEFT and measure the difference.

    Constructive test of the minimum fix. Editing from the end of the document
    backwards means every edit lands after the region touched next, so no
    offset is ever invalidated.

    Uses a generic [LABEL] placeholder rather than the pipeline's pools: the
    fix concerns WHERE replacements land, not what they are, and residual-PII
    measurement does not depend on the substitute's content.
    """
    import random
    rng = random.Random(seed)
    leaked = total = 0
    ctx_total = ctx_lost = 0
    docs_high = 0
    high_leaked = high_total = 0

    for d in docs:
        orig = d["original_text"]
        gold_sorted = sorted(d["gold"], key=lambda s: s["start"])
        text = orig
        for span in sorted(d["gold"], key=lambda s: s["start"], reverse=True):
            if rng.random() > coverage:
                continue
            text = text[:span["start"]] + f"[{span['label']}]" + text[span["end"]:]

        doc_high = False
        for span in gold_sorted:
            if len(span["value"]) < MIN_LEN:
                continue
            total += 1
            tier = SEVERITY.get(span["label"], MEDIUM)
            if tier == HIGH:
                high_total += 1
            if span["value"] in text:
                leaked += 1
                if tier == HIGH:
                    high_leaked += 1
                    doc_high = True
        docs_high += doc_high

        cuts = ([0] + [c for s in gold_sorted for c in (s["start"], s["end"])]
                + [len(orig)])
        for i in range(0, len(cuts) - 1, 2):
            seg = orig[cuts[i]:cuts[i + 1]]
            if len(seg) < MIN_CONTEXT:
                continue
            ctx_total += 1
            if seg not in text:
                ctx_lost += 1

    print("\nMINIMUM FIX — same replacements, applied right to left")
    print(f"  residual PII            {leaked:5d} / {total:5d}  "
          f"{leaked / total:6.1%}   (currently 19.1%)")
    print(f"  HIGH-severity residual  {high_leaked:5d} / {high_total:5d}  "
          f"{high_leaked / high_total:6.1%}   (currently 19.1%)")
    print(f"  docs with HIGH leak     {docs_high:5d} / {len(docs):5d}  "
          f"{docs_high / len(docs):6.1%}   (currently 20.6%)")
    print(f"  non-PII context lost    {ctx_lost:5d} / {ctx_total:5d}  "
          f"{ctx_lost / ctx_total:6.1%}   (currently 38.1%)")


def prove_mechanism(docs):
    """Identify the defect from the output alone, by exact prediction.

    The released text reveals which labels were rendered as a literal [LABEL]
    placeholder. For documents where every span used one AND nothing was
    dropped, the redacted text is fully predictable without knowing any
    replacement pool — so both splice orders can be tested against reality.
    """
    import re
    placeholder_labels = set()
    for d in docs:
        placeholder_labels.update(re.findall(r"\[([A-Z]+)\]", d["redacted_text"]))

    def splice(text, spans, reverse):
        for s in sorted(spans, key=lambda x: x["start"], reverse=reverse):
            text = text[:s["start"]] + f"[{s['label']}]" + text[s["end"]:]
        return text

    exact_lr = exact_rl = n = 0
    for d in docs:
        labels = [s["label"] for s in d["gold"]]
        if not labels or not all(x in placeholder_labels for x in labels):
            continue
        if d["spans_replaced"] != len(d["gold"]):
            continue
        n += 1
        exact_lr += splice(d["original_text"], d["gold"], False) == d["redacted_text"]
        exact_rl += splice(d["original_text"], d["gold"], True) == d["redacted_text"]

    print("\nMECHANISM PROOF — predicting the released text from the source")
    print(f"  fully-predictable documents          {n:5d}")
    print(f"  left-to-right reproduces exactly     {exact_lr:5d} ({exact_lr / n:.1%})")
    print(f"  right-to-left reproduces exactly     {exact_rl:5d} ({exact_rl / n:.1%})")


if __name__ == "__main__":
    docs = load_joined()
    verify_echo(docs)
    show_pairs(docs, n=6)
    residual_pii(docs)
    replacement_integrity(docs)
    prove_mechanism(docs)
    minimum_fix_simulation(docs)
