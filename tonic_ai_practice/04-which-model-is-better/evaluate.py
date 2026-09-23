"""Evaluation harness for problem 04 — Model A vs Model B.

Alignment machinery is carried over from 01-detector: greedy one-to-one
matching ranked by (exact, same_label, overlap). Kept as a copy rather than a
shared import because `01-detector` is not a legal module name.

Everything is measured from the three JSONL files. Nothing is justified by
reading build_models.py — see the contamination note in PROBLEM.md.
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
GOLD_PATH = DATA_DIR / "train_en_2500.jsonl"
PRED_PATHS = {"A": DATA_DIR / "predictions_a.jsonl",
              "B": DATA_DIR / "predictions_b.jsonl"}


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def load_all():
    """Gold spans plus both models' predictions, grouped by document."""
    rows = read_jsonl(GOLD_PATH)
    docs = {}
    for r in rows:
        did = str(r["id"])
        assert did not in docs, f"duplicate document id {did}"
        docs[did] = {
            "doc_id": did,
            "text": r["source_text"],
            "gold": [{"doc_id": did, "start": s["start"], "end": s["end"],
                      "label": s["label"], "text": s["value"]}
                     for s in r["privacy_mask"]],
            "pred": {"A": [], "B": []},
        }

    bad_join = bad_range = 0
    for name, path in PRED_PATHS.items():
        for p in read_jsonl(path):
            d = docs.get(p["doc_id"])
            if d is None:
                bad_join += 1
                continue
            if not (0 <= p["start"] < p["end"] <= len(d["text"])):
                bad_range += 1
                continue
            d["pred"][name].append({
                "doc_id": p["doc_id"], "start": p["start"], "end": p["end"],
                "label": p["label"], "score": p["score"],
                "text": d["text"][p["start"]:p["end"]],
            })

    # gold offsets must slice back to gold values, or every number is noise
    bad_offsets = sum(
        1 for d in docs.values() for s in d["gold"]
        if d["text"][s["start"]:s["end"]] != s["text"]
    )

    n_gold = sum(len(d["gold"]) for d in docs.values())
    print(f"documents {len(docs)}   gold spans {n_gold}")
    for name in PRED_PATHS:
        print(f"  model {name}: {sum(len(d['pred'][name]) for d in docs.values())} spans")
    print(f"  predictions with unknown doc_id : {bad_join}")
    print(f"  predictions with invalid offsets: {bad_range}")
    print(f"  gold offset mismatches          : {bad_offsets}")
    assert bad_join == 0 and bad_range == 0 and bad_offsets == 0

    gold_labels = {s["label"] for d in docs.values() for s in d["gold"]}
    for name in PRED_PATHS:
        pl = {s["label"] for d in docs.values() for s in d["pred"][name]}
        extra = pl - gold_labels
        print(f"  model {name} labels: {len(pl)} of {len(gold_labels)} gold types"
              f"{'  UNKNOWN: ' + str(extra) if extra else ''}")
    return docs


def overlap(a, b):
    return max(0, min(a["end"], b["end"]) - max(a["start"], b["start"]))


def align(gold, pred, label_strict=True):
    """Greedy one-to-one alignment, best pairs first.

    Ranked by (exact, same_label, overlap): a prediction straddling two gold
    spans claims the one it agrees with, not the one it merely covers most.
    With label_strict=False every overlap counts as a label agreement, which
    scores offsets only — the right view for a redaction use.
    """
    cands = []
    for gi, g in enumerate(gold):
        for pi, p in enumerate(pred):
            ov = overlap(g, p)
            if ov == 0:
                continue
            same = True if not label_strict else g["label"] == p["label"]
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


def prf(tp, n_pred, n_gold):
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_gold if n_gold else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def score_model(docs, model, label_strict=True):
    """Micro P/R/F1 under exact and relaxed matching, plus per-label counts."""
    exact_tp = relaxed_tp = n_pred = n_gold = 0
    per_label = defaultdict(lambda: Counter())

    for d in docs.values():
        gold, pred = d["gold"], d["pred"][model]
        n_gold += len(gold)
        n_pred += len(pred)
        for g in gold:
            per_label[g["label"]]["gold"] += 1

        # A matched prediction counts toward the GOLD label's precision
        # denominator, not its own. Under label-blind matching the two can
        # differ, and crediting the TP to one label while counting the
        # prediction under another makes per-label precision incoherent.
        matched_p = set()
        for gi, pi, is_exact, same in align(gold, pred, label_strict):
            if same:
                relaxed_tp += 1
                lab = gold[gi]["label"]
                per_label[lab]["relaxed_tp"] += 1
                per_label[lab]["pred"] += 1
                matched_p.add(pi)
                if is_exact:
                    exact_tp += 1
                    per_label[lab]["exact_tp"] += 1
        for pi, p in enumerate(pred):
            if pi not in matched_p:      # a false positive for its own type
                per_label[p["label"]]["pred"] += 1

    return {
        "exact": prf(exact_tp, n_pred, n_gold),
        "relaxed": prf(relaxed_tp, n_pred, n_gold),
        "n_pred": n_pred, "n_gold": n_gold, "per_label": per_label,
    }


def macro_f1(per_label, mode="relaxed"):
    key = f"{mode}_tp"
    scores = []
    for _lab, c in per_label.items():
        if not c["gold"]:
            continue
        scores.append(prf(c[key], c["pred"], c["gold"])[2])
    return sum(scores) / len(scores) if scores else 0.0


def show_docs(docs, n=8, width=150):
    for d in list(docs.values())[:n]:
        print(f"\n{'=' * 78}\ndoc {d['doc_id']}")
        print(f"  TEXT: {d['text'][:width]}")
        for name in ("gold", "A", "B"):
            spans = d["gold"] if name == "gold" else d["pred"][name]
            items = [(s["label"], s["text"])
                     for s in sorted(spans, key=lambda x: x["start"])]
            print(f"  {name:>4}: {items[:7]}")


def step1(docs):
    print("\n" + "=" * 70)
    print("STEP 1 — does B's F1 advantage reproduce?")
    print("=" * 70)

    for strict in (True, False):
        title = "LABEL-STRICT (label must match)" if strict else \
                "LABEL-BLIND (offsets only — the redaction view)"
        print(f"\n{title}")
        print(f"{'model':>6s} {'spans':>7s} "
              f"{'P_exact':>8s} {'R_exact':>8s} {'F1_exact':>9s}   "
              f"{'P_rel':>7s} {'R_rel':>7s} {'F1_rel':>8s} {'F1_macro':>9s}")
        res = {}
        for m in ("A", "B"):
            r = score_model(docs, m, label_strict=strict)
            res[m] = r
            pe, re_, fe = r["exact"]
            pr, rr, fr = r["relaxed"]
            print(f"{m:>6s} {r['n_pred']:7d} "
                  f"{pe:8.4f} {re_:8.4f} {fe:9.4f}   "
                  f"{pr:7.4f} {rr:7.4f} {fr:8.4f} "
                  f"{macro_f1(r['per_label']):9.4f}")
        for mode in ("exact", "relaxed"):
            gap = res["B"][mode][2] - res["A"][mode][2]
            print(f"   gap ({mode:7s}) B - A = {gap:+.4f} "
                  f"({gap * 100:+.2f} points)  "
                  f"{'B ahead' if gap > 0 else 'A ahead'}")
        gm = macro_f1(res["B"]["per_label"]) - macro_f1(res["A"]["per_label"])
        print(f"   gap (macro  ) B - A = {gm:+.4f} ({gm * 100:+.2f} points)  "
              f"{'B ahead' if gm > 0 else 'A ahead'}")




def _doc_stats(docs, model, label_strict, labels):
    """Per-document, per-label (tp, pred, gold). Alignment runs once."""
    import numpy as np
    idx = {lab: i for i, lab in enumerate(labels)}
    D, L = len(docs), len(labels)
    tp = np.zeros((D, L)); pr = np.zeros((D, L)); gd = np.zeros((D, L))

    for di, d in enumerate(docs.values()):
        gold, pred = d["gold"], d["pred"][model]
        for g in gold:
            gd[di, idx[g["label"]]] += 1
        matched = set()
        for gi, pi, _e, same in align(gold, pred, label_strict):
            if same:
                li = idx[gold[gi]["label"]]
                tp[di, li] += 1
                pr[di, li] += 1
                matched.add(pi)
        for pi, p in enumerate(pred):
            if pi not in matched:
                pr[di, idx[p["label"]]] += 1
    return tp, pr, gd


def _micro(tp, pr, gd):
    t, p, g = tp.sum(), pr.sum(), gd.sum()
    return 2 * t / (p + g) if (p + g) else 0.0


def _macro(tp, pr, gd):
    import numpy as np
    keep = gd > 0
    denom = pr + gd
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
    return f1[keep].mean() if keep.any() else 0.0


def bootstrap_gap(docs, label_strict=True, n_boot=2000, seed=0):
    """Paired bootstrap CI on the F1 difference, B - A.

    Documents are the independent unit, so documents are resampled. Both
    models are scored on the SAME resample each iteration — discarding the
    pairing would inflate the interval, since the two models agree on most
    documents.
    """
    import numpy as np
    labels = sorted({s["label"] for d in docs.values() for s in d["gold"]})
    stats = {m: _doc_stats(docs, m, label_strict, labels) for m in ("A", "B")}

    def gaps_for(fn, sample):
        out = []
        for m in ("A", "B"):
            tp, pr, gd = stats[m]
            out.append(fn(tp[sample].sum(0), pr[sample].sum(0), gd[sample].sum(0)))
        return out[1] - out[0]

    rng = np.random.default_rng(seed)
    D = len(docs)
    all_docs = np.arange(D)
    obs_micro = gaps_for(_micro, all_docs)
    obs_macro = gaps_for(_macro, all_docs)

    gm = np.empty(n_boot); gM = np.empty(n_boot)
    for i in range(n_boot):
        s = rng.integers(0, D, D)
        gm[i] = gaps_for(_micro, s)
        gM[i] = gaps_for(_macro, s)

    tag = "LABEL-STRICT" if label_strict else "LABEL-BLIND"
    print(f"\n{tag} — paired bootstrap, {n_boot} resamples of {D} documents")
    for name, obs, arr in (("micro F1", obs_micro, gm), ("macro F1", obs_macro, gM)):
        lo, hi = np.percentile(arr, [2.5, 97.5])
        crosses = lo <= 0 <= hi
        print(f"  {name}  gap B-A = {obs * 100:+6.2f} pts   "
              f"95% CI [{lo * 100:+6.2f}, {hi * 100:+6.2f}]   "
              f"width {(hi - lo) * 100:5.2f}   "
              f"{'CROSSES ZERO' if crosses else 'excludes zero'}   "
              f"B<=A in {(arr <= 0).mean():.1%}")




def counts(docs, model, label_strict):
    """Integer TP / FP / FN under relaxed matching."""
    tp = n_pred = n_gold = 0
    for d in docs.values():
        gold, pred = d["gold"], d["pred"][model]
        n_gold += len(gold)
        n_pred += len(pred)
        tp += sum(1 for _gi, _pi, _e, same in align(gold, pred, label_strict) if same)
    return {"tp": tp, "fp": n_pred - tp, "fn": n_gold - tp,
            "n_pred": n_pred, "n_gold": n_gold}


def f_beta(c, beta):
    """beta > 1 weights recall more; beta < 1 weights precision more."""
    p = c["tp"] / c["n_pred"] if c["n_pred"] else 0.0
    r = c["tp"] / c["n_gold"] if c["n_gold"] else 0.0
    b2 = beta * beta
    return (1 + b2) * p * r / (b2 * p + r) if (b2 * p + r) else 0.0


def step4(docs):
    print("\n" + "=" * 70)
    print("STEP 4 — does the ranking survive a cost model that isn't 1:1?")
    print("=" * 70)

    for strict in (True, False):
        tag = "LABEL-STRICT" if strict else "LABEL-BLIND"
        ca, cb = counts(docs, "A", strict), counts(docs, "B", strict)
        pa, ra = ca["tp"] / ca["n_pred"], ca["tp"] / ca["n_gold"]
        pb, rb = cb["tp"] / cb["n_pred"], cb["tp"] / cb["n_gold"]

        print(f"\n{tag}")
        print(f"  {'':6s} {'TP':>6s} {'FP':>6s} {'FN':>6s} "
              f"{'precision':>10s} {'recall':>8s}")
        print(f"  {'A':6s} {ca['tp']:6d} {ca['fp']:6d} {ca['fn']:6d} "
              f"{pa:10.4f} {ra:8.4f}")
        print(f"  {'B':6s} {cb['tp']:6d} {cb['fp']:6d} {cb['fn']:6d} "
              f"{pb:10.4f} {rb:8.4f}")

        if pa >= pb and ra >= rb:
            print("  -> A DOMINATES: higher precision AND higher recall. "
                  "No cost weighting can prefer B.")
        elif pb >= pa and rb >= ra:
            print("  -> B DOMINATES: higher precision AND higher recall. "
                  "No cost weighting can prefer A.")
        else:
            print("  -> trade-off: neither dominates, so the cost model decides.")
            # cost = r*FN + FP ; solve for the ratio r where the two tie
            dfn = ca["fn"] - cb["fn"]
            dfp = cb["fp"] - ca["fp"]
            if dfn != 0:
                r_star = dfp / dfn
                print(f"     tie when a MISS costs {r_star:.2f}x a FALSE POSITIVE")
                print(f"     miss cheaper than {r_star:.2f}x -> A wins;  "
                      f"dearer -> B wins")

        print(f"  {'beta':>6s} {'F_beta A':>10s} {'F_beta B':>10s} {'winner':>8s}")
        prev = None
        for beta in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0):
            fa, fb = f_beta(ca, beta), f_beta(cb, beta)
            w = "A" if fa > fb else "B"
            flip = "  <- flips here" if prev and w != prev else ""
            print(f"  {beta:6.2f} {fa:10.4f} {fb:10.4f} {w:>8s}{flip}")
            prev = w




# Severity tiers carried over from 01-detector, unchanged.
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


def outcome_split(docs, model):
    """Three-way split of predictions, and misses by severity tier.

    A prediction that overlaps gold but carries the wrong type is NOT the same
    failure as one that fires on empty text: the entity still gets redacted, so
    it costs nothing on privacy and costs a wrong-typed replacement on utility.
    Collapsing the two is what made the strict and blind panels incomparable.
    """
    from collections import Counter
    tp = type_confused = spurious = 0
    fn_by_tier = Counter()
    gold_by_tier = Counter()

    for d in docs.values():
        gold, pred = d["gold"], d["pred"][model]
        # label-blind alignment finds every prediction that located an entity
        pairs = align(gold, pred, label_strict=False)
        matched_g = {gi for gi, _pi, _e, _s in pairs}
        for gi, pi, _e, _s in pairs:
            if gold[gi]["label"] == pred[pi]["label"]:
                tp += 1
            else:
                type_confused += 1
        spurious += len(pred) - len(pairs)
        for gi, g in enumerate(gold):
            tier = SEVERITY.get(g["label"], MEDIUM)
            gold_by_tier[tier] += 1
            if gi not in matched_g:
                fn_by_tier[tier] += 1

    return {"tp": tp, "type_confused": type_confused, "spurious": spurious,
            "fn_by_tier": fn_by_tier, "gold_by_tier": gold_by_tier,
            "fn_total": sum(fn_by_tier.values())}


def step4b(docs):
    print("\n" + "=" * 70)
    print("STEP 4b — FP split, and severity-weighted tie point")
    print("=" * 70)

    r = {m: outcome_split(docs, m) for m in ("A", "B")}

    print(f"\nPrediction outcomes (offsets-based, so the two panels reconcile)")
    print(f"  {'':6s} {'located+typed':>14s} {'located,wrong type':>19s} {'spurious':>10s}")
    for m in ("A", "B"):
        print(f"  {m:6s} {r[m]['tp']:14d} {r[m]['type_confused']:19d} "
              f"{r[m]['spurious']:10d}")
    print("  note: strict FP = spurious + type_confused; blind FP = spurious")

    print(f"\nMisses by severity tier (entity never located at all)")
    print(f"  {'tier':8s} {'gold':>7s} {'FN A':>7s} {'FN B':>7s} "
          f"{'rate A':>8s} {'rate B':>8s}")
    for tier in (HIGH, MEDIUM, LOW):
        g = r["A"]["gold_by_tier"][tier]
        fa, fb = r["A"]["fn_by_tier"][tier], r["B"]["fn_by_tier"][tier]
        print(f"  {tier:8s} {g:7d} {fa:7d} {fb:7d} {fa / g:8.1%} {fb / g:8.1%}")

    print(f"\nTie point: cost = r x (missed entities) + 1 x (spurious spans)")
    d_sp = r["B"]["spurious"] - r["A"]["spurious"]
    for tier_name, fa, fb in (
        ("all severities", r["A"]["fn_total"], r["B"]["fn_total"]),
        ("HIGH only", r["A"]["fn_by_tier"][HIGH], r["B"]["fn_by_tier"][HIGH]),
    ):
        d_fn = fa - fb
        star = d_sp / d_fn if d_fn else float("inf")
        print(f"  {tier_name:16s} FN A={fa:5d}  FN B={fb:5d}  "
              f"extra spurious from B={d_sp:5d}  ->  tie at r = {star:.2f}")


if __name__ == "__main__":
    docs = load_all()
    show_docs(docs, n=6)
    step1(docs)
    print("\n" + "=" * 70)
    print("STEP 2 — is either gap bigger than measurement noise?")
    print("=" * 70)
    bootstrap_gap(docs, label_strict=True)
    bootstrap_gap(docs, label_strict=False)
    step4(docs)
    step4b(docs)
