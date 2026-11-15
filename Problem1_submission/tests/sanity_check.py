"""Synthetic sanity check for the detector.

No real backdoor traces are available (by design of the challenge), so
this script builds a small *synthetic* proxy anomaly set by perturbing
the relationship between counters in real clean traces (more LLC
pressure and fewer cycles for the same work - a plausible shape for a
triggered branch doing extra memory accesses). This is only a sanity
check that the detector reacts to distributional shifts in the counter
relationships; it is NOT a substitute for the private validation set
and is not used anywhere in training or threshold selection.
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "trace.csv"
OUT_PATH = ROOT / "tests" / "_synthetic_eval.csv"


def build_synthetic_eval(n: int = 100, seed: int = 0) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    df = pd.read_csv(DATA_PATH)

    clean_sample = df.sample(n, random_state=rng.randint(1e6)).reset_index(drop=True)
    clean_sample["label"] = 0

    anom = df.sample(n, random_state=rng.randint(1e6)).reset_index(drop=True).copy()
    anom["LLC-loads"] = anom["LLC-loads"] * 1.35 + 150000
    anom["cycles"] = anom["cycles"] * 0.92
    anom["label"] = 1

    return pd.concat([clean_sample, anom], ignore_index=True)


if __name__ == "__main__":
    test_df = build_synthetic_eval()
    test_df.to_csv(OUT_PATH, index=False)
    print(f"Wrote synthetic sanity-check set to {OUT_PATH}")
    print("Run: python ../src/detect.py _synthetic_eval.csv")
