"""Evaluation harness for problem 05 — the fidelity report.

Everything is measured from the three CSV files. Nothing is justified by
reading synthesize.py — see the contamination note in PROBLEM.md.
"""

from pathlib import Path

import pandas as pd
from scipy.stats import ks_2samp

DATA_DIR = Path(__file__).parent / "data"
PATHS = {
    "train": DATA_DIR / "real_train.csv",
    "holdout": DATA_DIR / "real_holdout.csv",
    "synth": DATA_DIR / "synthetic.csv",
}


def load():
    """Three frames, schema-aligned, with the control validated."""
    frames = {k: pd.read_csv(p) for k, p in PATHS.items()}

    cols = list(frames["train"].columns)
    for name, df in frames.items():
        assert list(df.columns) == cols, f"{name} column order differs"
    print(f"columns aligned: {len(cols)}")
    for name, df in frames.items():
        print(f"  {name:8s} {df.shape}")

    numeric = [c for c in cols if pd.api.types.is_numeric_dtype(frames["train"][c])]
    categorical = [c for c in cols if c not in numeric]
    print(f"  numeric     ({len(numeric)}): {numeric}")
    print(f"  categorical ({len(categorical)}): {categorical}")

    return frames, cols, numeric, categorical


def row_keys(df, cols):
    """Hashable row identity over the given columns.

    Built by vectorised concatenation with an elementwise `str`, not
    `astype(str).agg(join)`: on some pandas versions `astype(str)` leaves NaN
    as a float and the join raises. `.map(str)` renders every value, missing
    ones included, and is far faster than a row-wise apply.
    """
    out = df[cols[0]].map(str)
    for c in cols[1:]:
        out = out + "\x1f" + df[c].map(str)
    return out


def verify_control(frames, cols):
    """Is the holdout genuinely disjoint from train, by row VALUE?

    The split was by index, but Adult contains duplicate records, so identical
    rows can appear in both. Any such overlap is the rate at which two
    unrelated real samples coincide — the baseline every later privacy claim
    is measured against. If it is large, the control is weak and must be said.
    """
    tr = set(row_keys(frames["train"], cols))
    ho = row_keys(frames["holdout"], cols)
    overlap = ho.isin(tr).sum()
    print("\nCONTROL — holdout rows that also appear verbatim in train:")
    print(f"  {overlap} of {len(ho)}  ({overlap / len(ho):.3%})")
    if overlap == 0:
        print("  control is clean: any synthetic/train exact match is not chance")
    else:
        print("  baseline is non-zero; Step 2 must compare against this rate")
    return overlap / len(ho)


def total_variation(a, b):
    """0.5 * sum |p_a - p_b| over the union of categories. NaN is a category."""
    pa = a.value_counts(normalize=True, dropna=False)
    pb = b.value_counts(normalize=True, dropna=False)
    idx = pa.index.union(pb.index)
    return 0.5 * (pa.reindex(idx, fill_value=0) - pb.reindex(idx, fill_value=0)).abs().sum()


def step1(frames, numeric, categorical, claim=0.02):
    """Reproduce the fidelity report, with a real-vs-real control column."""
    print("\n" + "=" * 72)
    print("STEP 1 — does the green fidelity report reproduce?")
    print("=" * 72)
    print("  'vs holdout' is the control: two REAL samples of the same")
    print("  population, so it shows what natural sampling variation looks like.")

    print("\nNUMERIC — two-sample KS statistic")
    print(f"  {'column':<18}{'synth vs train':>16}{'holdout vs train':>19}  flag")
    worst_ks = worst_tv = 0.0
    for c in numeric:
        d_syn = ks_2samp(frames["synth"][c].dropna(),
                         frames["train"][c].dropna()).statistic
        d_ho = ks_2samp(frames["holdout"][c].dropna(),
                        frames["train"][c].dropna()).statistic
        worst_ks = max(worst_ks, d_syn)
        flag = "" if d_syn < claim else "  OVER CLAIM"
        tight = "  <- tighter than real-vs-real" if d_syn < d_ho else ""
        print(f"  {c:<18}{d_syn:>16.4f}{d_ho:>19.4f}{flag}{tight}")

    print("\nCATEGORICAL — total variation distance")
    print(f"  {'column':<18}{'synth vs train':>16}{'holdout vs train':>19}  flag")
    for c in categorical:
        d_syn = total_variation(frames["synth"][c], frames["train"][c])
        d_ho = total_variation(frames["holdout"][c], frames["train"][c])
        worst_tv = max(worst_tv, d_syn)
        flag = "" if d_syn < claim else "  OVER CLAIM"
        tight = "  <- tighter than real-vs-real" if d_syn < d_ho else ""
        print(f"  {c:<18}{d_syn:>16.4f}{d_ho:>19.4f}{flag}{tight}")

    # Report the two statistics separately. The claim was about KS, which is
    # defined only on the numeric columns; TV covers the categoricals. TV is
    # the stricter of the two (KS <= TV for any ordering), so holding TV to a
    # KS-derived threshold is conservative, not unfair.
    print(f"\n  worst KS  (6 numeric columns)    : {worst_ks:.4f}  "
          f"-> claim 'all under {claim}' reproduces: {worst_ks < claim}")
    print(f"  worst TV  (9 categorical columns): {worst_tv:.4f}  "
          f"-> under {claim}: {worst_tv < claim}")
    print(f"  the fidelity report covers {len(numeric)} of "
          f"{len(numeric) + len(categorical)} columns")
    return worst_ks, worst_tv


def category_loss(frames, categorical, show=4):
    """Which categorical values exist in the real data but not the synthetic.

    A distance statistic says a column drifted; it does not say the column lost
    half its vocabulary. This separates the two.
    """
    print("\nCATEGORY COVERAGE — values present in real train vs in synthetic")
    print(f"  {'column':<18}{'real':>7}{'synth':>7}{'lost':>7}")
    worst = None
    for c in categorical:
        r = frames["train"][c].nunique(dropna=False)
        s = frames["synth"][c].nunique(dropna=False)
        print(f"  {c:<18}{r:>7}{s:>7}{r - s:>7}")
        if worst is None or (r - s) > worst[1]:
            worst = (c, r - s)

    col = worst[0]
    pr = frames["train"][col].value_counts(normalize=True, dropna=False)
    ps = frames["synth"][col].value_counts(normalize=True, dropna=False)
    missing = set(pr.index) - set(ps.index)
    print(f"\n  worst column: {col} — {len(missing)} values absent from synthetic")
    print(f"  {'value':<22}{'real share':>12}{'synth share':>13}{'delta':>10}")
    for k in list(pr.index[:show]):
        print(f"  {str(k):<22}{pr[k]:>12.4f}{ps.get(k, 0):>13.4f}"
              f"{ps.get(k, 0) - pr[k]:>+10.4f}")
    lost_mass = sum(pr[k] for k in missing)
    print(f"  total real share carried by the absent values: {lost_mass:.4f}")
    return col, len(missing), lost_mass


def exact_match(frames, cols, show=2):
    """Step 2 — how many synthetic rows are verbatim copies of real rows?

    Exact-match rate: the share of synthetic rows identical in every column to
    some real training row. Compared against the control — the rate at which an
    unrelated REAL sample (the holdout) matches the same training set by chance.
    Both use train as the reference set, so the rates are directly comparable.
    """
    print("\n" + "=" * 72)
    print("STEP 2 — verbatim copies of real rows")
    print("=" * 72)

    variants = [("all 15 columns", cols),
                ("14 columns, fnlwgt excluded", [c for c in cols if c != "fnlwgt"])]
    results = {}
    for name, use in variants:
        tr_keys = row_keys(frames["train"], use)
        tr = set(tr_keys)
        ho = set(row_keys(frames["holdout"], use))
        sy = row_keys(frames["synth"], use)
        ho_in_tr = row_keys(frames["holdout"], use).isin(tr).mean()
        sy_in_tr = sy.isin(tr)
        sy_in_ho = sy.isin(ho).mean()

        # of the matched training rows, how many are unique in the training set?
        # a unique row points to exactly one real person.
        counts = tr_keys.value_counts()
        matched = sy[sy_in_tr]
        unique_hits = matched.map(counts).eq(1).sum()
        distinct_people = matched[matched.map(counts).eq(1)].nunique()
        # the same measure for the control: chance matches should mostly land
        # on COMMON profiles, not on rows that belong to a single real person
        ho_all = row_keys(frames["holdout"], use)
        ho_matched = ho_all[ho_all.isin(tr)]
        ho_unique_rate = ho_matched.map(counts).eq(1).sum() / len(ho_all)
        sy_unique_rate = unique_hits / len(sy)

        ratio = sy_in_tr.mean() / ho_in_tr if ho_in_tr else float("inf")
        print(f"\n  matching on {name}")
        print(f"    control: holdout rows found in train   {ho_in_tr:8.3%}")
        print(f"    synthetic rows found in train          {sy_in_tr.mean():8.3%}"
              f"   ({sy_in_tr.sum()} rows)   {ratio:,.2f}x the control")
        print(f"    synthetic rows found in holdout        {sy_in_ho:8.3%}"
              f"   (holdout is 1/3 the size; not directly comparable)")
        print(f"    matched rows unique in train           {unique_hits}"
              f"   -> {distinct_people} distinct real individuals")
        u_ratio = sy_unique_rate / ho_unique_rate if ho_unique_rate else float("inf")
        print(f"    rows matching a ONE-PERSON train profile: "
              f"synthetic {sy_unique_rate:.3%}  vs control {ho_unique_rate:.3%}"
              f"   ({u_ratio:,.1f}x)")
        results[name] = (sy_in_tr, unique_hits, distinct_people, ratio)

    # one concrete example a non-statistician can check by eye
    sy_in_tr = results["all 15 columns"][0]
    tr_keys = row_keys(frames["train"], cols)
    counts = tr_keys.value_counts()
    sy_keys = row_keys(frames["synth"], cols)
    hits = sy_in_tr[sy_in_tr].index
    unique_hits = [i for i in hits if counts[sy_keys[i]] == 1]
    print("\n  EXAMPLES — synthetic row next to the real training row it copies")
    for i in unique_hits[:show]:
        j = tr_keys[tr_keys == sy_keys[i]].index[0]
        print(f"\n    synthetic row {i}  <->  real training row {j}")
        both = pd.DataFrame({"synthetic": frames["synth"].loc[i],
                             "real": frames["train"].loc[j]})
        for c, r in both.iterrows():
            print(f"      {c:<16} {str(r['synthetic']):<22} {str(r['real'])}")
    return results


QI = ["age", "sex", "race", "marital-status", "education", "native-country"]
SENSITIVE = ["occupation", "class", "capital-gain", "hours-per-week"]


def linkage_attack(frames, show=1):
    """Step 4 — can an attacker who knows six facts about someone learn more?

    The attacker knows a target's quasi-identifiers (QI) and looks them up in
    the synthetic file. If exactly one synthetic row carries that QI combination,
    they read off its sensitive attributes. Disclosure = all of those equal the
    target's real values.

    Run against TRAINING people (the generator saw them) and HOLDOUT people (it
    did not). A safe file leaks no more about the first group than the second.
    """
    print("\n" + "=" * 72)
    print("STEP 4 — quasi-identifier linkage")
    print("=" * 72)
    print(f"  QI (attacker knows):     {QI}")
    print(f"  sensitive (to recover):  {SENSITIVE}")

    tr_q = row_keys(frames["train"], QI)
    k = tr_q.map(tr_q.value_counts())
    print("\n  k-anonymity of the REAL training data on these QIs:")
    print(f"    k = 1  (unique — one person) : {(k == 1).mean():7.2%}")
    print(f"    k <= 5                       : {(k <= 5).mean():7.2%}")

    sy_q = row_keys(frames["synth"], QI)
    sy_s = row_keys(frames["synth"], SENSITIVE)
    singletons = sy_q[sy_q.map(sy_q.value_counts()) == 1]
    lookup = dict(zip(singletons, sy_s[singletons.index]))

    def attack(targets):
        tq = row_keys(targets, QI)
        ts = row_keys(targets, SENSITIVE)
        linked = tq.isin(lookup.keys())
        hit = [i for i, (q, s) in enumerate(zip(tq, ts)) if lookup.get(q) == s]
        return linked.mean(), len(hit) / len(targets), len(hit), hit

    res = {}
    print(f"\n  {'targets':<28}{'uniquely linked':>17}{'fully disclosed':>18}")
    for name, key in (("training people (seen)", "train"),
                      ("holdout people (unseen)", "holdout")):
        linked, rate, n, hit = attack(frames[key])
        res[key] = (rate, n, hit)
        print(f"  {name:<28}{linked:>17.2%}{rate:>12.2%} ({n})")

    ratio = res["train"][0] / res["holdout"][0] if res["holdout"][0] else float("inf")
    print(f"\n  disclosure rate, seen vs unseen: {ratio:,.1f}x")

    # Is this a second leak, or the Step 2 copies seen through another lens?
    # Mark which training people have their full record copied verbatim.
    all_cols = list(frames["train"].columns)
    copied = row_keys(frames["train"], all_cols).isin(
        set(row_keys(frames["synth"], all_cols)))
    hits = res["train"][2]
    in_copies = sum(1 for i in hits if copied.iloc[i])
    excess = res["train"][1] - res["holdout"][0] * len(frames["train"])
    print(f"\n  of the {len(hits)} disclosed training people:")
    print(f"    also verbatim-copied in Step 2 : {in_copies}")
    print(f"    not copied                     : {len(hits) - in_copies}")
    print(f"  expected by chance at the holdout rate: "
          f"{res['holdout'][0] * len(frames['train']):.0f}  "
          f"(excess over chance: {excess:.0f})")

    for i in res["train"][2][:show]:
        t = frames["train"].iloc[i]
        print("\n  EXAMPLE — attacker knows: " +
              ", ".join(f"{c}={t[c]}" for c in QI))
        print("            synthetic file reveals: " +
              ", ".join(f"{c}={t[c]}" for c in SENSITIVE) + "   (all correct)")
    return res


def tstr(frames, target="class", seed=0):
    """Step 6 — train on synthetic, test on real (TSTR).

    Fit the same classifier on different training sets, always scored on the
    real holdout the generator never saw:
      real train        -> the ceiling (TRTR: train real, test real)
      synthetic         -> what the data science team would actually get
      synthetic minus the verbatim copies -> the synthetic part on its own
      only the 600 copied real rows       -> what the leaked rows alone buy
    ROC AUC is the headline because income is imbalanced (~24% >50K), so plain
    accuracy rewards always predicting the majority class.
    """
    import numpy as np
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.preprocessing import OrdinalEncoder

    print("\n" + "=" * 72)
    print("STEP 6 — train on synthetic, test on real")
    print("=" * 72)

    feats = [c for c in frames["train"].columns if c != target]
    cats = [c for c in feats if not pd.api.types.is_numeric_dtype(frames["train"][c])]
    positive = ">50K"
    test = frames["holdout"]
    y_test = (test[target] == positive).astype(int)
    majority = max(y_test.mean(), 1 - y_test.mean())

    # which synthetic rows are verbatim copies of a training row (Step 2)
    all_cols = list(frames["train"].columns)
    tr_keys = set(row_keys(frames["train"], all_cols))
    is_copy = row_keys(frames["synth"], all_cols).isin(tr_keys)
    copied_real = frames["train"][row_keys(frames["train"], all_cols)
                                  .isin(set(row_keys(frames["synth"], all_cols)))]

    def fit_score(train_df):
        # encoder fitted on the training set only; categories it never saw
        # (e.g. nationalities absent from the synthetic file) become missing
        enc = OrdinalEncoder(handle_unknown="use_encoded_value",
                             unknown_value=np.nan, encoded_missing_value=np.nan)
        Xtr = train_df[feats].copy()
        Xte = test[feats].copy()
        Xtr[cats] = enc.fit_transform(Xtr[cats].astype(str))
        Xte[cats] = enc.transform(Xte[cats].astype(str))
        ytr = (train_df[target] == positive).astype(int)
        model = HistGradientBoostingClassifier(
            categorical_features=[feats.index(c) for c in cats], random_state=seed)
        model.fit(Xtr, ytr)
        p = model.predict_proba(Xte)[:, 1]
        return roc_auc_score(y_test, p), accuracy_score(y_test, p >= 0.5), p

    def auc_ci(p, n_boot=2000):
        # bootstrap the HOLDOUT people: resample with replacement, rescore the
        # same fitted model, take the 2.5th and 97.5th percentiles of AUC
        rng = np.random.default_rng(seed)
        y = y_test.to_numpy()
        n = len(y)
        aucs = np.empty(n_boot)
        for b in range(n_boot):
            idx = rng.integers(0, n, n)
            aucs[b] = roc_auc_score(y[idx], p[idx])
        return np.percentile(aucs, [2.5, 97.5]), aucs.std()

    runs = [
        ("real train (ceiling)", frames["train"]),
        ("synthetic, as delivered", frames["synth"]),
        ("synthetic minus verbatim copies", frames["synth"][~is_copy]),
        ("only the copied real rows", copied_real),
    ]
    print(f"  scored on {len(test)} real holdout rows; "
          f"majority-class accuracy = {majority:.3f}; random AUC = 0.500")
    print(f"\n  {'trained on':<34}{'rows':>7}{'AUC':>8}{'95% CI':>18}{'SE':>7}{'acc':>7}")
    out = {}
    for name, df in runs:
        auc, acc, p = fit_score(df)
        (lo, hi), se = auc_ci(p)
        out[name] = (auc, acc, len(df), lo, hi)
        note = "   contains 0.5 -> indistinguishable from random" if lo <= 0.5 <= hi else ""
        print(f"  {name:<34}{len(df):>7}{auc:>8.3f}   [{lo:.3f}, {hi:.3f}]"
              f"{se:>7.3f}{acc:>7.3f}{note}")
    return out


if __name__ == "__main__":
    frames, cols, numeric, categorical = load()
    verify_control(frames, cols)
    step1(frames, numeric, categorical)
    category_loss(frames, categorical)
    exact_match(frames, cols)
    linkage_attack(frames)
    tstr(frames)
