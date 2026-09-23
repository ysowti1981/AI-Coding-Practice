"""Download AG News, sample train and test, cache as JSONL.

Run once before the clock starts. Deliberately prints no content — the data
is not inspected here.
"""

import json
from pathlib import Path

from datasets import load_dataset

DATASET = "ag_news"
SEED = 0

# AG News ships integer labels; these are the documented names, in order.
LABEL_NAMES = ["World", "Sports", "Business", "Sci/Tech"]

SPLITS = {"train": 4000, "test": 2000}

DATA_DIR = Path(__file__).parent / "data"
HF_CACHE_DIR = DATA_DIR / "hf_cache"


def out_path(split, n):
    return DATA_DIR / f"{split}_{n}.jsonl"


def _sample(ds, n, seed=SEED):
    """Tag every row with its original index, then shuffle and take n."""
    ds = ds.map(lambda _row, i: {"orig_idx": i}, with_indices=True)
    return ds.shuffle(seed=seed).select(range(min(n, len(ds))))


def download(force=False):
    """Cache one JSONL per split. No-op for a split already on disk."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    written = {}

    for split, n in SPLITS.items():
        path = out_path(split, n)
        if path.exists() and not force:
            print(f"already cached: {path.name}")
            written[split] = path
            continue

        ds = load_dataset(DATASET, split=split, cache_dir=str(HF_CACHE_DIR))
        sample = _sample(ds, n)

        with open(path, "w") as f:
            for row in sample:
                f.write(json.dumps({
                    "doc_id": f"{split}-{row['orig_idx']}",
                    "text": row["text"],
                    "label": row["label"],
                    "label_name": LABEL_NAMES[row["label"]],
                }) + "\n")

        print(f"wrote {len(sample)} rows to {path.name} (of {len(ds)} available)")
        written[split] = path

    return written


if __name__ == "__main__":
    download()
