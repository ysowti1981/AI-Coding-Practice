"""Span-level NER evaluation: gold (data/) vs model predictions (preds/).

Conventions (both reported):
  strict  - a prediction counts only if (start, end, label) equal a gold span exactly.
  lenient - same label and any character overlap, matched greedily 1:1,
            largest overlap first (so one long prediction cannot claim two gold spans).
The unit is the entity span; the bootstrap resamples documents, because spans
inside one document are not independent.
"""
import argparse
import json
import os
from collections import Counter

import numpy as np
import pandas as pd

LABELS = ["name", "drug_name", "drug_amount", "medical_condition", "date"]
SEED = 0
N_BOOT = 1000


# ---------------------------------------------------------------- loading + checks

def load_jsonl(path):
    """One document per line -> list of dicts."""
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def overlaps(a, b):
    """True if two end-exclusive character spans share at least one character."""
    return a["start"] < b["end"] and b["start"] < a["end"]


def check_integrity(name, gold, pred):
    """Verify the join is total and offsets round-trip before any metric is computed.

    Raises on anything that would silently corrupt the metric (missing/extra ids,
    duplicate ids, different texts, spans whose offsets don't reproduce their text).
    Overlapping gold spans are only reported: they are legal but matter for matching.
    """
    g_ids = [d["id"] for d in gold]
    p_ids = [d["id"] for d in pred]
    assert len(set(g_ids)) == len(g_ids), f"{name}: duplicate gold ids"
    assert len(set(p_ids)) == len(p_ids), f"{name}: duplicate pred ids"
    missing = sorted(set(g_ids) - set(p_ids))
    extra = sorted(set(p_ids) - set(g_ids))

    p_by_id = {d["id"]: d for d in pred}
    text_diff, bad_offsets, gold_overlaps = 0, 0, 0
    n_gold = n_pred = 0
    for g in gold:
        p = p_by_id.get(g["id"])
        if p is None:
            continue
        if g["text"] != p["text"]:
            text_diff += 1
        for side, doc in (("gold", g), ("pred", p)):
            for s in doc["spans"]:
                if doc["text"][s["start"]:s["end"]] != s["text"]:
                    bad_offsets += 1
            if side == "gold":
                n_gold += len(doc["spans"])
            else:
                n_pred += len(doc["spans"])
        spans = sorted(g["spans"], key=lambda s: s["start"])
        gold_overlaps += sum(overlaps(a, b) for a, b in zip(spans, spans[1:]))

    labels_seen = {s["label"] for d in gold + pred for s in d["spans"]}
    print(f"[integrity] {name}: docs gold={len(gold)} pred={len(pred)} "
          f"missing={len(missing)} extra={len(extra)} text_diff={text_diff} "
          f"bad_offsets={bad_offsets} gold_spans={n_gold} pred_spans={n_pred} "
          f"overlapping_gold_pairs={gold_overlaps}")
    if missing or extra or text_diff or bad_offsets:
        raise ValueError(f"{name}: join not total / offsets broken "
                         f"(missing={missing[:5]} extra={extra[:5]})")
    unknown = labels_seen - set(LABELS)
    assert not unknown, f"{name}: unexpected labels {unknown}"


# ---------------------------------------------------------------- matching

def match_strict(gold_spans, pred_spans):
    """Return set of (gold_idx, pred_idx) pairs with identical (start, end, label)."""
    key_to_pred = {}
    for j, p in enumerate(pred_spans):
        key_to_pred.setdefault((p["start"], p["end"], p["label"]), []).append(j)
    pairs = set()
    for i, g in enumerate(gold_spans):
        js = key_to_pred.get((g["start"], g["end"], g["label"]))
        if js:
            pairs.add((i, js.pop()))  # pop: each prediction can match once
    return pairs


def match_lenient(gold_spans, pred_spans):
    """Same label + any overlap, greedy 1:1, largest character overlap first."""
    cands = []
    for i, g in enumerate(gold_spans):
        for j, p in enumerate(pred_spans):
            if g["label"] == p["label"] and overlaps(g, p):
                ov = min(g["end"], p["end"]) - max(g["start"], p["start"])
                cands.append((-ov, i, j))
    cands.sort()
    used_g, used_p, pairs = set(), set(), set()
    for _, i, j in cands:
        if i not in used_g and j not in used_p:
            used_g.add(i)
            used_p.add(j)
            pairs.add((i, j))
    return pairs


def doc_counts(gold_spans, pred_spans, matcher):
    """Per-label [tp, fp, fn] array for one document, shape (n_labels, 3)."""
    pairs = matcher(gold_spans, pred_spans)
    counts = np.zeros((len(LABELS), 3), dtype=int)
    for i, _ in pairs:
        counts[LABELS.index(gold_spans[i]["label"]), 0] += 1
    for g in gold_spans:
        counts[LABELS.index(g["label"]), 2] += 1
    for p in pred_spans:
        counts[LABELS.index(p["label"]), 1] += 1
    counts[:, 1] -= counts[:, 0]  # fp = preds - tp
    counts[:, 2] -= counts[:, 0]  # fn = gold - tp
    return counts


def prf(tp, fp, fn):
    """Precision, recall, F1; 0 when undefined (no preds / no gold)."""
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def micro_f1(counts):
    """Micro F1 from a (n_labels, 3) array: pool all spans, every span weighs the same."""
    tp, fp, fn = counts.sum(axis=0)
    return prf(tp, fp, fn)[2]


# ---------------------------------------------------------------- metrics

def score_dataset(name, gold, pred, matcher, convention):
    """Rows of per-label + micro + macro metrics, and the per-doc count tensor."""
    p_by_id = {d["id"]: d for d in pred}
    per_doc = np.stack([doc_counts(g["spans"], p_by_id[g["id"]]["spans"], matcher)
                        for g in gold])  # (n_docs, n_labels, 3)
    total = per_doc.sum(axis=0)

    n_gold = Counter(s["label"] for d in gold for s in d["spans"])
    n_pred = Counter(s["label"] for d in pred for s in d["spans"])
    rows = []
    for k, lab in enumerate(LABELS):
        tp, fp, fn = total[k]
        assert tp + fn == n_gold[lab], (name, lab, "tp+fn != gold")
        assert tp + fp == n_pred[lab], (name, lab, "tp+fp != pred")
        p, r, f = prf(tp, fp, fn)
        rows.append(dict(dataset=name, convention=convention, label=lab,
                         support=n_gold[lab], n_pred=n_pred[lab],
                         tp=tp, fp=fp, fn=fn, precision=p, recall=r, f1=f))
    tp, fp, fn = total.sum(axis=0)
    p, r, f = prf(tp, fp, fn)
    rows.append(dict(dataset=name, convention=convention, label="MICRO",
                     support=tp + fn, n_pred=tp + fp, tp=tp, fp=fp, fn=fn,
                     precision=p, recall=r, f1=f))
    # macro: unweighted mean over labels, so rare labels count as much as common ones
    lab_rows = rows[:len(LABELS)]
    rows.append(dict(dataset=name, convention=convention, label="MACRO",
                     support=tp + fn, n_pred=tp + fp, tp=np.nan, fp=np.nan, fn=np.nan,
                     precision=np.mean([x["precision"] for x in lab_rows]),
                     recall=np.mean([x["recall"] for x in lab_rows]),
                     f1=np.mean([x["f1"] for x in lab_rows])))
    return rows, per_doc


def bootstrap_ci(per_doc, n_boot=N_BOOT, seed=SEED):
    """95% percentile CIs for micro F1 and each label's F1, resampling documents.

    Documents are the independent unit: spans in one document share a writer,
    a format and a patient, so resampling spans would understate the noise.
    """
    rng = np.random.default_rng(seed)
    n = per_doc.shape[0]
    micro, per_label = [], []
    for _ in range(n_boot):
        c = per_doc[rng.integers(0, n, n)].sum(axis=0)
        micro.append(micro_f1(c))
        per_label.append([prf(*c[k])[2] for k in range(len(LABELS))])
    micro = np.percentile(micro, [2.5, 97.5])
    per_label = np.percentile(np.array(per_label), [2.5, 97.5], axis=0)
    cis = {"MICRO": tuple(micro)}
    for k, lab in enumerate(LABELS):
        cis[lab] = (per_label[0, k], per_label[1, k])
    return cis


# ---------------------------------------------------------------- error taxonomy

def classify_errors(name, gold, pred):
    """One row per gold span and per prediction that is not an exact match.

    Gold buckets (first that applies): exact / boundary (same-label pred overlaps,
    not exact) / label_confusion (only different-label preds overlap) / missed.
    Pred buckets for non-exact preds: boundary / label_confusion / spurious (no gold overlap).
    Existence-based, not 1:1: one long pred covering two gold spans makes both 'boundary'.
    """
    p_by_id = {d["id"]: d for d in pred}
    rows, confusion = [], Counter()
    for g_doc in gold:
        text = g_doc["text"]
        gs, ps = g_doc["spans"], p_by_id[g_doc["id"]]["spans"]
        exact_keys = {(p["start"], p["end"], p["label"]) for p in ps}
        gold_keys = {(g["start"], g["end"], g["label"]) for g in gs}

        for g in gs:
            ov = [p for p in ps if overlaps(g, p)]
            same = [p for p in ov if p["label"] == g["label"]]
            other = [p for p in ov if p["label"] != g["label"]]
            for p in ov:
                confusion[(g["label"], p["label"])] += 1
            if not ov:
                confusion[(g["label"], "O")] += 1
            best = None
            if (g["start"], g["end"], g["label"]) in exact_keys:
                cat = "exact"
            elif same:
                cat = "boundary"
                best = max(same, key=lambda p: min(g["end"], p["end"]) - max(g["start"], p["start"]))
            elif other:
                cat = "label_confusion"
                best = max(other, key=lambda p: min(g["end"], p["end"]) - max(g["start"], p["start"]))
            else:
                cat = "missed"
            rows.append(dict(dataset=name, id=g_doc["id"], side="gold", label=g["label"],
                             category=cat, other_label=best["label"] if best else "",
                             start=g["start"], end=g["end"], text=g["text"],
                             pred_start=best["start"] if best else -1,
                             pred_end=best["end"] if best else -1,
                             pred_text=best["text"] if best else "",
                             context=context(text, g["start"], g["end"])))

        for p in ps:
            ov = [g for g in gs if overlaps(g, p)]
            if not ov:
                confusion[("O", p["label"])] += 1
            if (p["start"], p["end"], p["label"]) in gold_keys:
                cat = "exact"
            elif any(g["label"] == p["label"] for g in ov):
                cat = "boundary"
            elif ov:
                cat = "label_confusion"
            else:
                cat = "spurious"
            rows.append(dict(dataset=name, id=g_doc["id"], side="pred", label=p["label"],
                             category=cat,
                             other_label=ov[0]["label"] if cat == "label_confusion" else "",
                             start=p["start"], end=p["end"], text=p["text"],
                             pred_start=p["start"], pred_end=p["end"], pred_text=p["text"],
                             context=context(text, p["start"], p["end"])))
    return rows, confusion


def context(text, start, end, width=40):
    """Single-line snippet with the span in [brackets] and newlines escaped."""
    s = text[max(0, start - width):start] + "[" + text[start:end] + "]" + text[end:end + width]
    return s.replace("\n", "\\n")


def boundary_direction(err, text_by_key):
    """For gold 'boundary' errors: is the pred longer, shorter or shifted, and by what text?"""
    b = err[(err.side == "gold") & (err.category == "boundary")]
    direction, extra, missing = Counter(), Counter(), Counter()
    for r in b.itertuples():
        text = text_by_key[(r.dataset, r.id)]
        gs, ge, ps, pe = r.start, r.end, r.pred_start, r.pred_end
        if ps <= gs and pe >= ge:
            direction["pred_longer"] += 1
        elif ps >= gs and pe <= ge:
            direction["pred_shorter"] += 1
        else:
            direction["shifted"] += 1
        # characters the pred adds (left/right) or drops (left/right)
        if ps < gs:
            extra["L:" + repr(text[ps:gs])] += 1
        if pe > ge:
            extra["R:" + repr(text[ge:pe])] += 1
        if ps > gs:
            missing["L:" + repr(text[gs:ps])] += 1
        if pe < ge:
            missing["R:" + repr(text[pe:ge])] += 1
    return direction, extra, missing


# ---------------------------------------------------------------- printing

def print_examples(err, n):
    """N real false negatives and N real false positives per label (strict sense)."""
    for (ds, lab), grp in err.groupby(["dataset", "label"], sort=False):
        fn = grp[(grp.side == "gold") & (grp.category != "exact")].head(n)
        fp = grp[(grp.side == "pred") & (grp.category != "exact")].head(n)
        print(f"\n--- {ds} / {lab}")
        for r in fn.itertuples():
            extra = f" pred={r.pred_text!r}({r.other_label or r.label})" if r.pred_text else ""
            print(f"  FN {r.category:15s} {r.id}: {r.context}{extra}")
        for r in fp.itertuples():
            extra = f" gold_label={r.other_label}" if r.other_label else ""
            print(f"  FP {r.category:15s} {r.id}: {r.context}{extra}")


def has_digit(s):
    return any(ch.isdigit() for ch in s)


def print_amount_format(err, datasets, n=3):
    """drug_amount surface form: does the span contain a digit, and does recall depend on it?

    Train is the baseline (gold only, there are no train preds). For evaluated sets,
    recall is strict (gold span counted as found only if category == 'exact'),
    split by digit / no digit - if the no-digit rows carry the misses, the format is the cause.
    """
    train = load_jsonl("data/train.jsonl")
    tr = [(d["id"], s, d["text"]) for d in train for s in d["spans"] if s["label"] == "drug_amount"]
    n_dig = sum(has_digit(s["text"]) for _, s, _ in tr)
    print(f"  {'train':16s} spans={len(tr):5d}  with digit={n_dig / len(tr):6.1%}  (gold only, no preds)")
    rows = []
    for ds in datasets:
        g = err[(err.dataset == ds) & (err.side == "gold") & (err.label == "drug_amount")]
        dig = g.text.map(has_digit)
        found = g.category == "exact"
        rec = lambda mask: f"{found[mask].mean():.3f} (n={mask.sum()})" if mask.sum() else "  -   (n=0)"
        print(f"  {ds:16s} spans={len(g):5d}  with digit={dig.mean():6.1%}  "
              f"recall|digit={rec(dig)}  recall|no digit={rec(~dig)}")
        rows.append((ds, g))

    # Train transcripts are the like-for-like comparison for vendor A transcripts ('trn-t-' ids).
    print(f"\n  examples - train transcripts (gold in [ ]):")
    for doc_id, s, text in [t for t in tr if t[0].startswith("trn-t-")][:n]:
        print(f"    {doc_id}: {context(text, s['start'], s['end'], 30)}")
    for ds, g in rows:
        print(f"  examples - {ds} (gold in [ ], then what the model predicted there):")
        for r in g.head(n).itertuples():
            pred = f"{r.other_label or 'drug_amount'}:{r.pred_text!r}" if r.pred_text else \
                ("exact match" if r.category == "exact" else "NOTHING")
            print(f"    {r.id}: {r.context} -> {pred}")


def main(datasets, n_examples):
    os.makedirs("results", exist_ok=True)
    metric_rows, error_rows, text_by_key, confusions = [], [], {}, {}
    for ds in datasets:
        gold = load_jsonl(f"data/{ds}.jsonl")
        pred = load_jsonl(f"preds/{ds}.jsonl")
        check_integrity(ds, gold, pred)
        for d in gold:
            text_by_key[(ds, d["id"])] = d["text"]

        strict_rows, per_doc = score_dataset(ds, gold, pred, match_strict, "strict")
        lenient_rows, _ = score_dataset(ds, gold, pred, match_lenient, "lenient")
        cis = bootstrap_ci(per_doc)
        for r in strict_rows:
            r["ci_lo"], r["ci_hi"] = cis.get(r["label"], (np.nan, np.nan))
        metric_rows += strict_rows + lenient_rows

        rows, conf = classify_errors(ds, gold, pred)
        error_rows += rows
        confusions[ds] = conf

    m = pd.DataFrame(metric_rows)
    err = pd.DataFrame(error_rows)
    m.to_csv("results/metrics.csv", index=False)
    err.to_csv("results/errors.csv", index=False)

    # reconcile: taxonomy buckets must sum to the same totals as the metric tables
    for ds in datasets:
        for side in ("gold", "pred"):
            n_rows = len(err[(err.dataset == ds) & (err.side == side)])
            micro = m[(m.dataset == ds) & (m.convention == "strict") & (m.label == "MICRO")].iloc[0]
            expected = micro.support if side == "gold" else micro.n_pred
            assert n_rows == expected, (ds, side, n_rows, expected)
            n_exact = len(err[(err.dataset == ds) & (err.side == side) & (err.category == "exact")])
            assert n_exact == micro.tp, (ds, side, "exact != strict tp", n_exact, micro.tp)

    pd.set_option("display.width", 200)
    print("\n== Headline: micro F1 (strict with 95% doc-bootstrap CI, seed=0, 1000 reps)")
    s = m[(m.convention == "strict") & (m.label == "MICRO")].set_index("dataset")
    l = m[(m.convention == "lenient") & (m.label == "MICRO")].set_index("dataset")
    head = pd.DataFrame({
        "docs": [sum(k[0] == ds for k in text_by_key) for ds in s.index],
        "gold": s.support, "pred": s.n_pred,
        "strict_F1": s.f1.round(3),
        "95%CI": [f"[{a:.3f}, {b:.3f}]" for a, b in zip(s.ci_lo, s.ci_hi)],
        "lenient_F1": l.f1.round(3),
        "strict_macro": m[(m.convention == "strict") & (m.label == "MACRO")].set_index("dataset").f1.round(3),
        "lenient_macro": m[(m.convention == "lenient") & (m.label == "MACRO")].set_index("dataset").f1.round(3),
    })
    print(head.to_string())

    print("\n== Per-label F1 (strict / lenient), support = gold spans")
    per = m[m.label.isin(LABELS)]
    tab = per.pivot_table(index=["dataset", "label"], columns="convention",
                          values="f1", sort=False).round(3)
    st = per[per.convention == "strict"].set_index(["dataset", "label"])
    tab["P_strict"] = st.precision.round(3)
    tab["R_strict"] = st.recall.round(3)
    tab["support"] = st.support
    tab["n_pred"] = st.n_pred
    tab["strict_CI"] = [f"[{a:.2f}, {b:.2f}]" for a, b in zip(st.ci_lo, st.ci_hi)]
    print(tab[["support", "n_pred", "strict", "strict_CI", "lenient", "P_strict", "R_strict"]].to_string())

    print("\n== Error taxonomy, gold spans (rows sum to support)")
    g = err[err.side == "gold"]
    gt = pd.crosstab([g.dataset, g.label], g.category).reindex(
        columns=["exact", "boundary", "label_confusion", "missed"], fill_value=0)
    gt["total"] = gt.sum(axis=1)
    print(gt.loc[datasets].to_string())

    print("\n== Error taxonomy, predicted spans (rows sum to n_pred)")
    p = err[err.side == "pred"]
    pt = pd.crosstab([p.dataset, p.label], p.category).reindex(
        columns=["exact", "boundary", "label_confusion", "spurious"], fill_value=0)
    pt["total"] = pt.sum(axis=1)
    print(pt.loc[datasets].to_string())

    print("\n== Boundary-only errors: direction and the text added/dropped (top 5)")
    for ds in datasets:
        for lab in LABELS:
            sub = err[(err.dataset == ds) & (err.label == lab)]
            d, ex, mi = boundary_direction(sub, text_by_key)
            if sum(d.values()) == 0:
                continue
            print(f"  {ds:16s} {lab:18s} {dict(d)}")
            print(f"      extra:   {ex.most_common(5)}")
            print(f"      missing: {mi.most_common(5)}")

    print("\n== Label confusion (gold x pred over overlapping pairs; O = no overlap)")
    for ds in datasets:
        c = confusions[ds]
        rows_ = LABELS + ["O"]
        cm = pd.DataFrame([[c.get((a, b), 0) for b in rows_] for a in rows_],
                          index=["g:" + r for r in rows_], columns=["p:" + r for r in rows_])
        print(f"  {ds}")
        print(cm.to_string())

    print("\n== drug_amount format: train vs evaluated sets (digits vs spelled-out numbers)")
    print_amount_format(err, datasets)

    if n_examples:
        print_examples(err, n_examples)
    print("\nwrote results/metrics.csv, results/errors.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["val", "prospect_test_a", "prospect_test_b"])
    ap.add_argument("--examples", type=int, default=0)
    args = ap.parse_args()
    main(args.datasets, args.examples)
