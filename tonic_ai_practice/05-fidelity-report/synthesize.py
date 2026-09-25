"""Generate synthetic data from the real training set.

Run once to produce data/synthetic.csv.
"""

import numpy as np
import pandas as pd

from load_data import DATA_DIR, TRAIN_PATH

SEED = 42
RARE_THRESHOLD = 0.01     # categorical values below this get collapsed
COPY_FRACTION = 0.04

OUT_PATH = DATA_DIR / "synthetic.csv"


def main():
    real = pd.read_csv(TRAIN_PATH)
    n = len(real)
    rng = np.random.default_rng(SEED)

    # 1. sample each column independently from its empirical distribution
    synth = pd.DataFrame({
        col: real[col].to_numpy()[rng.integers(0, n, n)]
        for col in real.columns
    })

    # 2. collapse rare categorical values into the most common one
    for col in synth.columns:
        if synth[col].dtype == object:
            freq = synth[col].value_counts(normalize=True)
            rare = freq[freq < RARE_THRESHOLD].index
            if len(rare) > 0:
                synth.loc[synth[col].isin(rare), col] = freq.idxmax()

    # 3. replace a fraction of rows with real rows
    k = int(COPY_FRACTION * n)
    target = rng.choice(n, size=k, replace=False)
    source = rng.choice(n, size=k, replace=False)
    synth.iloc[target] = real.iloc[source].to_numpy()

    synth = synth[real.columns]
    synth.to_csv(OUT_PATH, index=False)
    print(f"wrote {len(synth)} rows x {synth.shape[1]} columns to {OUT_PATH.name}")


if __name__ == "__main__":
    main()
