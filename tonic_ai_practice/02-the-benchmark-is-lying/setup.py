"""Generate the two prediction files being evaluated.

Run ONCE before the clock starts, then treat the outputs as opaque — in the
real interview these arrive as files from someone else. Prints counts only:
no metrics, because computing them is the first move of the timed hour.

Model A: TF-IDF word 1-2gram      + LogisticRegression  -> score = max P(class)
Model B: TF-IDF char_wb 3-5gram   + LinearSVC           -> score = decision margin

Two deliberately different inductive biases, so the errors are not identical.
"""

import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

from load_data import DATA_DIR, LABEL_NAMES, SEED, out_path

TRAIN_PATH = out_path("train", 4000)
TEST_PATH = out_path("test", 2000)


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def build_models():
    """Two pipelines with genuinely different views of the text."""
    model_a = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True),
        LogisticRegression(max_iter=1000, random_state=SEED),
    )
    model_b = make_pipeline(
        TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True),
        LinearSVC(random_state=SEED),
    )
    return model_a, model_b


def write_predictions(path, doc_ids, pred_idx, scores):
    with open(path, "w") as f:
        for doc_id, p, s in zip(doc_ids, pred_idx, scores):
            f.write(json.dumps({
                "doc_id": doc_id,
                "pred_label": LABEL_NAMES[int(p)],
                "score": float(s),
            }) + "\n")
    print(f"wrote {len(doc_ids)} predictions to {path.name}")


def main():
    train = read_jsonl(TRAIN_PATH)
    test = read_jsonl(TEST_PATH)
    X_train = [r["text"] for r in train]
    y_train = [r["label"] for r in train]
    X_test = [r["text"] for r in test]
    doc_ids = [r["doc_id"] for r in test]
    print(f"train={len(train)}  test={len(test)}")

    model_a, model_b = build_models()

    model_a.fit(X_train, y_train)
    proba = model_a.predict_proba(X_test)
    pred_a = proba.argmax(axis=1)
    # Confidence: probability assigned to the predicted class. Bounded [0, 1].
    score_a = proba.max(axis=1)
    write_predictions(DATA_DIR / "predictions_a.jsonl", doc_ids, pred_a, score_a)

    model_b.fit(X_train, y_train)
    margins = model_b.decision_function(X_test)
    pred_b = margins.argmax(axis=1)
    # Margin: the one-vs-rest decision value of the predicted class. UNBOUNDED,
    # and not comparable to A's probability without a transform. (The other
    # defensible reading of "margin" is top1 - top2; that is a different number.)
    score_b = margins.max(axis=1)
    write_predictions(DATA_DIR / "predictions_b.jsonl", doc_ids, pred_b, score_b)


if __name__ == "__main__":
    main()
