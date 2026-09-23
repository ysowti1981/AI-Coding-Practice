"""Generate the two prediction files under evaluation.

Run ONCE before the clock starts, then treat the outputs as opaque — in a real
engagement these arrive as files from whoever trained the models. Prints counts
only: no metrics, no content.

Model A "the incumbent"  — conservative, high precision.
Model B "the challenger" — aggressive, high recall.
"""

import json
import random
import re

from load_data import OUT_PATH, DATA_DIR

SEED = 7

# --- Model A: conservative -------------------------------------------------
A_DROP_RATE = 0.18          # overall, but biased (see below)
A_HARD_TYPES = {"STREET", "ACCOUNTNUMBER"}   # dropped much harder
A_SPURIOUS_RATE = 0.03
A_BOUNDARY_RATE = 0.08      # biased toward UNDER-capture

# --- Model B: aggressive ---------------------------------------------------
B_DROP_RATE = 0.09          # roughly uniform
B_SPURIOUS_RATE = 0.14      # on capitalised / numeric non-entities
B_BOUNDARY_RATE = 0.08      # biased toward OVER-capture
B_WRONG_TYPE_RATE = 0.12    # right offsets, wrong label

CAP_TOKEN = re.compile(r"\b[A-Z][a-z]{2,}\b")
NUM_TOKEN = re.compile(r"\b\d{3,}\b")


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def _overlaps(a, b):
    return a[0] < b[1] and b[0] < a[1]


def spurious_candidates(text, gold):
    """Capitalised words and digit runs that are NOT gold entities."""
    taken = [(s["start"], s["end"]) for s in gold]
    out = []
    for pattern in (CAP_TOKEN, NUM_TOKEN):
        for m in pattern.finditer(text):
            here = (m.start(), m.end())
            if not any(_overlaps(here, t) for t in taken):
                out.append(here)
    return out


def _scaled_probs(raw, target):
    """Scale per-item probabilities so the mean lands on `target`."""
    total = sum(raw)
    if total == 0:
        return [target] * len(raw)
    scale = target * len(raw) / total
    return [min(p * scale, 0.95) for p in raw]


def build_a(rows, rng, labels):
    preds = []
    for row in rows:
        text, gold = row["source_text"], row["privacy_mask"]
        doc_id = str(row["id"])

        raw = []
        for s in gold:
            p = 1.0
            n = len(s["value"])
            if n > 12:
                p += 1.6
            elif n > 6:
                p += 0.7
            if s["label"] in A_HARD_TYPES:
                p += 3.5
            raw.append(p)
        probs = _scaled_probs(raw, A_DROP_RATE)

        for s, p_drop in zip(gold, probs):
            if rng.random() < p_drop:
                continue
            start, end, score = s["start"], s["end"], rng.uniform(0.62, 0.99)
            if rng.random() < A_BOUNDARY_RATE:
                # under-capture: shrink from one side
                if rng.random() < 0.5 and end - start > 2:
                    start += 1
                elif end - start > 2:
                    end -= 1
                score = rng.uniform(0.52, 0.88)
            preds.append((doc_id, start, end, s["label"], score))

        cands = spurious_candidates(text, gold)
        rng.shuffle(cands)
        for st, en in cands[:round(len(gold) * A_SPURIOUS_RATE + rng.random())]:
            preds.append((doc_id, st, en, rng.choice(labels),
                          rng.uniform(0.30, 0.72)))
    return preds


def build_b(rows, rng, labels):
    preds = []
    for row in rows:
        text, gold = row["source_text"], row["privacy_mask"]
        doc_id = str(row["id"])

        for s in gold:
            if rng.random() < B_DROP_RATE:
                continue
            start, end = s["start"], s["end"]
            label, score = s["label"], rng.uniform(0.60, 0.98)
            if rng.random() < B_BOUNDARY_RATE:
                # over-capture: spill outward
                start = max(0, start - rng.randint(1, 3))
                end = min(len(text), end + rng.randint(1, 3))
                score = rng.uniform(0.50, 0.90)
            if rng.random() < B_WRONG_TYPE_RATE:
                others = [x for x in labels if x != s["label"]]
                label = rng.choice(others)
                score = rng.uniform(0.50, 0.90)
            preds.append((doc_id, start, end, label, score))

        cands = spurious_candidates(text, gold)
        rng.shuffle(cands)
        for st, en in cands[:round(len(gold) * B_SPURIOUS_RATE + rng.random())]:
            preds.append((doc_id, st, en, rng.choice(labels),
                          rng.uniform(0.32, 0.80)))
    return preds


def write(preds, path):
    with open(path, "w") as f:
        for doc_id, start, end, label, score in preds:
            f.write(json.dumps({
                "doc_id": doc_id,
                "start": int(start),
                "end": int(end),
                "label": label,
                "score": round(float(score), 4),
            }) + "\n")
    print(f"wrote {len(preds)} predictions to {path.name}")


def main():
    rows = read_jsonl(OUT_PATH)
    labels = sorted({s["label"] for r in rows for s in r["privacy_mask"]})
    print(f"read {len(rows)} documents, {len(labels)} entity types")

    write(build_a(rows, random.Random(SEED), labels),
          DATA_DIR / "predictions_a.jsonl")
    write(build_b(rows, random.Random(SEED + 1), labels),
          DATA_DIR / "predictions_b.jsonl")


if __name__ == "__main__":
    main()
