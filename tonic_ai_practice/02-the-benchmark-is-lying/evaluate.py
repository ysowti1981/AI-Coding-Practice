"""Evaluation harness for problem 02.

Step 1: reproduce the reported numbers. Macro-F1, per-class F1, and the
confusion matrix for each model, from the prediction files alone.
"""

import json
import math
from collections import Counter

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score

from load_data import DATA_DIR, LABEL_NAMES, out_path

TEST_PATH = out_path("test", 2000)
PRED_PATHS = {
    "A": DATA_DIR / "predictions_a.jsonl",
    "B": DATA_DIR / "predictions_b.jsonl",
}


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def load_joined():
    """Gold labels plus each model's predictions, aligned by doc_id.

    The join is asserted total: every test document predicted exactly once by
    each model, nothing extra. A silently partial join would change every
    number below without failing.
    """
    gold = read_jsonl(TEST_PATH)
    gold_ids = [r["doc_id"] for r in gold]
    assert len(set(gold_ids)) == len(gold), "duplicate doc_id in the test set"

    y_true = [r["label_name"] for r in gold]
    preds = {}

    for name, path in PRED_PATHS.items():
        rows = read_jsonl(path)
        by_id = {r["doc_id"]: r for r in rows}
        assert len(by_id) == len(rows), f"duplicate doc_id in model {name}"
        assert set(by_id) == set(gold_ids), (
            f"model {name} does not cover the test set exactly: "
            f"{len(set(gold_ids) - set(by_id))} missing, "
            f"{len(set(by_id) - set(gold_ids))} extra"
        )
        preds[name] = [by_id[i]["pred_label"] for i in gold_ids]

    print(f"joined {len(y_true)} documents x {len(preds)} models, join is total")
    texts = [r["text"] for r in gold]
    return gold_ids, y_true, preds, texts


def print_confusion(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred, labels=LABEL_NAMES)
    print(f"\n{title}   (rows = true, cols = predicted)")
    print(f"{'':>10s}" + "".join(f"{l:>10s}" for l in LABEL_NAMES) + f"{'total':>8s}")
    for name, row in zip(LABEL_NAMES, cm):
        print(f"{name:>10s}" + "".join(f"{v:>10d}" for v in row) + f"{row.sum():>8d}")
    print(f"{'total':>10s}" + "".join(f"{v:>10d}" for v in cm.sum(axis=0)))


def mcnemar(y_true, pred_a, pred_b):
    """Exact McNemar on the discordant pairs.

    Same documents scored by both models, so the comparison is paired. Only
    the disagreements carry information: under the null that the models are
    equally accurate, each discordant pair is a fair coin flip, so the exact
    two-sided p-value is a binomial tail. No approximation, no scipy.

    NOTE: this tests per-item ACCURACY, not macro-F1. It answers "are these
    two distinguishable at all", which is not the same question as "is the
    macro-F1 gap real".
    """
    both = only_a = only_b = neither = 0
    for t, a, b in zip(y_true, pred_a, pred_b):
        ca, cb = a == t, b == t
        if ca and cb:
            both += 1
        elif ca:
            only_a += 1
        elif cb:
            only_b += 1
        else:
            neither += 1

    n = only_a + only_b
    k = min(only_a, only_b)
    # two-sided exact binomial: P(X <= k) + P(X >= n-k), symmetric at p=0.5
    tail = sum(math.comb(n, i) for i in range(k + 1)) * 0.5 ** n
    p = min(1.0, 2 * tail)

    print("\nMcNemar (paired, exact) — per-item accuracy")
    print(f"  both correct          {both:5d}")
    print(f"  only A correct        {only_a:5d}")
    print(f"  only B correct        {only_b:5d}")
    print(f"  both wrong            {neither:5d}")
    print(f"  discordant pairs      {n:5d}")
    print(f"  accuracy  A = {(both + only_a) / len(y_true):.4f}   "
          f"B = {(both + only_b) / len(y_true):.4f}")
    print(f"  exact two-sided p = {p:.4f}")
    return p


def _macro_f1_from_codes(t, p, n_classes=len(LABEL_NAMES)):
    """Macro-F1 straight from a bincount confusion matrix. Fast enough to
    run inside a bootstrap loop; a class absent from a resample scores 0."""
    cm = np.bincount(t * n_classes + p,
                     minlength=n_classes ** 2).reshape(n_classes, n_classes)
    tp = np.diag(cm).astype(float)
    denom = 2 * tp + (cm.sum(axis=0) - tp) + (cm.sum(axis=1) - tp)
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
    return f1.mean()


def bootstrap_gap(y_true, pred_a, pred_b, n_boot=10000, seed=0):
    """Percentile CI on the macro-F1 difference, B - A.

    Documents are the independent unit, so documents are what gets resampled.
    Both models are scored on the SAME resample each iteration — the pairing
    is the point. Resampling them independently would throw away the fact
    that they agree on most documents and inflate the interval.
    """
    code = {lab: i for i, lab in enumerate(LABEL_NAMES)}
    t = np.fromiter((code[x] for x in y_true), int, len(y_true))
    a = np.fromiter((code[x] for x in pred_a), int, len(pred_a))
    b = np.fromiter((code[x] for x in pred_b), int, len(pred_b))

    # the hand-rolled metric must agree with sklearn before it is trusted
    assert abs(_macro_f1_from_codes(t, a)
               - f1_score(y_true, pred_a, average="macro")) < 1e-12

    rng = np.random.default_rng(seed)
    n = len(t)
    gaps = np.empty(n_boot)
    for i in range(n_boot):
        s = rng.integers(0, n, n)
        gaps[i] = _macro_f1_from_codes(t[s], b[s]) - _macro_f1_from_codes(t[s], a[s])

    observed = _macro_f1_from_codes(t, b) - _macro_f1_from_codes(t, a)
    lo, hi = np.percentile(gaps, [2.5, 97.5])
    p_le_zero = float((gaps <= 0).mean())

    print(f"\nPaired bootstrap on macro-F1 gap (B - A), {n_boot} resamples")
    print(f"  observed gap        {observed:+.4f}  ({observed * 100:+.2f} points)")
    print(f"  95% CI             [{lo:+.4f}, {hi:+.4f}]  "
          f"([{lo * 100:+.2f}, {hi * 100:+.2f}] points)")
    print(f"  CI width            {(hi - lo) * 100:.2f} points  "
          f"({(hi - lo) / abs(observed):.1f}x the gap)")
    print(f"  resamples where B <= A   {p_le_zero:.1%}")
    print(f"  CI excludes zero?   {'YES' if lo > 0 else 'NO'}")
    return observed, lo, hi


AMBIGUOUS_PAIR = frozenset({"Business", "Sci/Tech"})


def label_quality_probe(y_true, pred_a, pred_b, texts, n_show=10):
    """Is B's advantage real corrections, or agreement with arbitrary labels?

    Every accuracy number assumes gold is right. Where a document is genuinely
    ambiguous, "correct" means matching whichever label the annotator picked,
    and a model that matches those coin flips more often scores higher without
    being better.

    The strongest available signal, short of re-annotating: in the both-wrong
    pile, the cases where A and B AGREE WITH EACH OTHER against gold. Two
    deliberately different inductive biases converging on the same answer is
    better evidence that the label is wrong than that both models erred.
    """
    only_a, only_b, both_wrong_agree, both_wrong_split = [], [], [], []
    for i, (t, a, b) in enumerate(zip(y_true, pred_a, pred_b)):
        ca, cb = a == t, b == t
        if ca and not cb:
            only_a.append(i)
        elif cb and not ca:
            only_b.append(i)
        elif not ca and not cb:
            (both_wrong_agree if a == b else both_wrong_split).append(i)

    def ambig_share(idxs, get_pred):
        hits = sum(1 for i in idxs if {y_true[i], get_pred(i)} == AMBIGUOUS_PAIR)
        return hits, (hits / len(idxs) if idxs else 0.0)

    print("\nWhere each model's wins live")
    # B wins when A was wrong: what confusion did B fix?
    n, share = ambig_share(only_b, lambda i: pred_a[i])
    print(f"  B wins (A wrong)        {len(only_b):4d}   "
          f"Business<->Sci/Tech: {n} ({share:.0%})")
    n, share = ambig_share(only_a, lambda i: pred_b[i])
    print(f"  A wins (B wrong)        {len(only_a):4d}   "
          f"Business<->Sci/Tech: {n} ({share:.0%})")

    print("\nThe both-wrong pile — where the benchmark's ceiling lives")
    n_bw = len(both_wrong_agree) + len(both_wrong_split)
    print(f"  both wrong                    {n_bw:4d}")
    print(f"    A and B agree with each other {len(both_wrong_agree):4d}  "
          f"<- label-error candidates")
    print(f"    A and B differ                {len(both_wrong_split):4d}")
    n, share = ambig_share(both_wrong_agree, lambda i: pred_a[i])
    print(f"  of the agreeing ones, Business<->Sci/Tech: {n} ({share:.0%})")

    counts = Counter((y_true[i], pred_a[i]) for i in both_wrong_agree)
    print("\n  gold -> what both models said instead (top 8)")
    for (t, p), c in counts.most_common(8):
        print(f"    {t:>9s} -> {p:<9s} {c:4d}")

    print(f"\n  {n_show} examples where BOTH models disagree with gold, "
          f"and agree with each other:")
    for i in both_wrong_agree[:n_show]:
        print(f"\n    gold={y_true[i]:<9s} both said={pred_a[i]}")
        print(f"    {texts[i][:200]}")

    return only_a, only_b, both_wrong_agree, both_wrong_split


def report(docs=None):
    _ids, y_true, preds, texts = load_joined()

    macro = {m: f1_score(y_true, p, average="macro") for m, p in preds.items()}
    gap = macro["B"] - macro["A"]

    print(f"\nmacro-F1   A = {macro['A']:.4f}   B = {macro['B']:.4f}   "
          f"gap = {gap:+.4f}  ({gap * 100:+.2f} points)")

    print(f"\nper-class F1")
    print(f"{'class':>10s} {'A':>8s} {'B':>8s} {'B - A':>8s} {'support':>8s}")
    fa = f1_score(y_true, preds["A"], average=None, labels=LABEL_NAMES)
    fb = f1_score(y_true, preds["B"], average=None, labels=LABEL_NAMES)
    for name, a, b in zip(LABEL_NAMES, fa, fb):
        support = sum(1 for t in y_true if t == name)
        print(f"{name:>10s} {a:8.4f} {b:8.4f} {b - a:+8.4f} {support:8d}")

    for model in ("A", "B"):
        print_confusion(y_true, preds[model], f"Model {model}")

    mcnemar(y_true, preds["A"], preds["B"])
    bootstrap_gap(y_true, preds["A"], preds["B"])
    label_quality_probe(y_true, preds["A"], preds["B"], texts)


if __name__ == "__main__":
    report()
