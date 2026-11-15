"""Feature engineering for HPC traces.

Raw HPC counters are supplemented with ratio features. A backdoor trigger
typically perturbs the *relationship* between counters (e.g. extra cache
pressure per cycle, or an unusual LLC-load-to-cache-reference ratio) even
when the raw counts stay within a plausible range individually. Ratios
make that relationship explicit for the detector.
"""

import numpy as np
import pandas as pd

RAW_COLUMNS = ["cache-references", "cycles", "LLC-loads"]

EPS = 1e-9


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in RAW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Input trace is missing expected columns: {missing}")

    out = df[RAW_COLUMNS].copy()
    out["cacheref_per_cycle"] = out["cache-references"] / (out["cycles"] + EPS)
    out["llcload_per_cycle"] = out["LLC-loads"] / (out["cycles"] + EPS)
    out["llcload_per_cacheref"] = out["LLC-loads"] / (out["cache-references"] + EPS)
    return out


FEATURE_COLUMNS = RAW_COLUMNS + [
    "cacheref_per_cycle",
    "llcload_per_cycle",
    "llcload_per_cacheref",
]
