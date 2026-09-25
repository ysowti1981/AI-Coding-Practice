"""Fetch the Adult census dataset, sample, split, cache as CSV.

Run once before the clock starts. Prints shapes only — the data is not
inspected here.
"""

import os
from pathlib import Path

# macOS python.org builds ship without root certificates, so the OpenML fetch
# fails SSL verification. Point OpenSSL at certifi's bundle rather than
# disabling verification. Must happen before the first TLS context is built.
try:
    import certifi
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
except ImportError:
    pass

from sklearn.datasets import fetch_openml  # noqa: E402

SEED = 0
N_SAMPLE = 20_000
N_TRAIN = 15_000          # the "real training" set the synthesizer sees
N_HOLDOUT = 5_000         # real rows the synthesizer never sees

DATA_DIR = Path(__file__).parent / "data"
TRAIN_PATH = DATA_DIR / "real_train.csv"
HOLDOUT_PATH = DATA_DIR / "real_holdout.csv"


def download(force=False):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if TRAIN_PATH.exists() and HOLDOUT_PATH.exists() and not force:
        print(f"already cached: {TRAIN_PATH.name}, {HOLDOUT_PATH.name}")
        return TRAIN_PATH, HOLDOUT_PATH

    adult = fetch_openml("adult", version=2, as_frame=True)
    df = adult.frame
    print(f"fetched {df.shape[0]} rows x {df.shape[1]} columns")

    sample = df.sample(n=N_SAMPLE, random_state=SEED).reset_index(drop=True)
    train = sample.iloc[:N_TRAIN]
    holdout = sample.iloc[N_TRAIN:N_TRAIN + N_HOLDOUT]

    train.to_csv(TRAIN_PATH, index=False)
    holdout.to_csv(HOLDOUT_PATH, index=False)
    print(f"wrote {len(train)} rows to {TRAIN_PATH.name}")
    print(f"wrote {len(holdout)} rows to {HOLDOUT_PATH.name}")
    return TRAIN_PATH, HOLDOUT_PATH


if __name__ == "__main__":
    download()
