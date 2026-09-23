"""Download AG-free PII corpus, filter to English, sample, cache as JSONL.

Adapted from 01-detector/load_data.py — same dataset and schema. Changes:
N_SAMPLE 1500 -> 2000, and a re-exported HF cache path. Kept as a copy
rather than a shared import because `01-detector` is not a legal module
name; a shared utils package would be the real fix.
"""

from pathlib import Path

from datasets import load_dataset

DATASET = "ai4privacy/pii-masking-200k"
SPLIT = "train"

N_SAMPLE = 2000
SEED = 0  # fixed so the sample is the same every run

# Column that carries the language, and the values that count as English.
# The 200k release spells it out ("English"); other releases use "en".
LANG_COLUMN = "language"
ENGLISH = {"english", "en"}

# Everything lands next to this script, not in ~/.cache, so the data is
# visible in the repo and gitignorable.
DATA_DIR = Path(__file__).parent / "data"
HF_CACHE_DIR = DATA_DIR / "hf_cache"
OUT_PATH = DATA_DIR / f"{SPLIT}_en_{N_SAMPLE}.jsonl"


def _filter_english(ds):
    """Keep English rows. Raise — loudly — rather than return an empty set."""
    if LANG_COLUMN not in ds.column_names:
        raise KeyError(
            f"no {LANG_COLUMN!r} column; columns are {ds.column_names}"
        )

    en = ds.filter(lambda row: str(row[LANG_COLUMN]).strip().lower() in ENGLISH)
    if len(en) == 0:
        seen = sorted(set(ds[LANG_COLUMN]))
        raise ValueError(
            f"language filter matched 0 of {len(ds)} rows; "
            f"values present: {seen}"
        )
    return en


def download(force: bool = False) -> Path:
    """Download once, filter to English, sample, cache as JSONL."""
    if OUT_PATH.exists() and not force:
        print(f"already cached: {OUT_PATH}")
        return OUT_PATH

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(DATASET, split=SPLIT, cache_dir=str(HF_CACHE_DIR))
    print(f"downloaded {len(ds)} rows, columns: {ds.column_names}")

    en = _filter_english(ds)
    print(f"english: {len(en)} rows")

    n = min(N_SAMPLE, len(en))
    if n < N_SAMPLE:
        print(f"only {n} english rows available, taking all of them")
    sample = en.shuffle(seed=SEED).select(range(n))

    sample.to_json(OUT_PATH)  # one JSON object per line
    print(f"wrote {len(sample)} rows to {OUT_PATH}")
    return OUT_PATH


if __name__ == "__main__":
    download()
