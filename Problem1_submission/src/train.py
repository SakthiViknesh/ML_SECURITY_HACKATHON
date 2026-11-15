"""Train an unsupervised HPC-trace anomaly detector for backdoor detection.

Only clean (benign) traces are available, so this is a strict one-class /
novelty-detection problem: the detector learns the shape of the clean
distribution and flags traces that fall outside it.

Method
------
An ensemble of two complementary one-class models is used:

1. Isolation Forest - captures non-linear / local structure in the
   feature space (good at isolating points in sparse regions).
2. Elliptic Envelope (robust Mahalanobis distance) - captures the global
   covariance structure of the clean traces and is sensitive to traces
   that are off the main correlation axis between counters, even if each
   individual counter value looks "in range".

Each model's raw score is min-max normalized (using statistics from the
training fold) and the two normalized scores are averaged into a single
ensemble anomaly score. The decision threshold is picked with 5-fold
cross-validation: in each fold the models are fit on the training rows
and scored on the held-out rows, the held-out scores are pooled across
all folds, and the threshold is the (1 - target_fpr) quantile of that
pooled, purely-clean distribution. This gives an honest estimate of the
threshold needed to hold a chosen false-positive rate on unseen clean
traces, without ever seeing a backdoor trace.

The final deployed models are refit on the full dataset (more data to
describe the clean manifold); the threshold found via cross-validation is
kept as-is since it already reflects held-out generalization.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler

from features import FEATURE_COLUMNS, engineer_features

RANDOM_STATE = 42
TARGET_FPR = 0.02  # tolerate ~2% false positives on clean traces
N_FOLDS = 5

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "trace.csv"
MODEL_PATH = ROOT / "model" / "detector.joblib"
METRICS_PATH = ROOT / "model" / "training_report.json"


def _fit_models(X: np.ndarray):
    iso = IsolationForest(
        n_estimators=200,
        contamination="auto",
        random_state=RANDOM_STATE,
    ).fit(X)
    ee = EllipticEnvelope(
        contamination=0.05,
        random_state=RANDOM_STATE,
    ).fit(X)
    return iso, ee


def _raw_scores(iso: IsolationForest, ee: EllipticEnvelope, X: np.ndarray):
    # Higher = more anomalous for both, after sign flips.
    iso_score = -iso.score_samples(X)
    ee_score = -ee.score_samples(X)
    return iso_score, ee_score


def _normalize(score: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip((score - lo) / (hi - lo + 1e-12), 0.0, 1.0)


def cross_validated_threshold(X: np.ndarray) -> tuple[float, dict]:
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    pooled_ensemble_scores = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        iso, ee = _fit_models(X_train)

        iso_train, ee_train = _raw_scores(iso, ee, X_train)
        iso_lo, iso_hi = iso_train.min(), iso_train.max()
        ee_lo, ee_hi = ee_train.min(), ee_train.max()

        iso_val, ee_val = _raw_scores(iso, ee, X_val)
        ensemble_val = 0.5 * _normalize(iso_val, iso_lo, iso_hi) + 0.5 * _normalize(
            ee_val, ee_lo, ee_hi
        )
        pooled_ensemble_scores.append(ensemble_val)

    pooled = np.concatenate(pooled_ensemble_scores)
    threshold = float(np.quantile(pooled, 1 - TARGET_FPR))
    achieved_fpr = float((pooled >= threshold).mean())
    stats = {
        "cv_folds": N_FOLDS,
        "target_fpr": TARGET_FPR,
        "achieved_fpr_on_pooled_held_out_clean": achieved_fpr,
        "pooled_score_mean": float(pooled.mean()),
        "pooled_score_std": float(pooled.std()),
    }
    return threshold, stats


def main():
    df = pd.read_csv(DATA_PATH)
    feat_df = engineer_features(df)
    X_raw = feat_df[FEATURE_COLUMNS].to_numpy()

    scaler = RobustScaler().fit(X_raw)
    X = scaler.transform(X_raw)

    threshold, cv_stats = cross_validated_threshold(X)

    # Final deployed models, fit on the full clean dataset.
    iso_final, ee_final = _fit_models(X)
    iso_full, ee_full = _raw_scores(iso_final, ee_final, X)
    norm_stats = {
        "iso_lo": float(iso_full.min()),
        "iso_hi": float(iso_full.max()),
        "ee_lo": float(ee_full.min()),
        "ee_hi": float(ee_full.max()),
    }

    bundle = {
        "scaler": scaler,
        "iso_forest": iso_final,
        "elliptic_envelope": ee_final,
        "feature_columns": FEATURE_COLUMNS,
        "norm_stats": norm_stats,
        "threshold": threshold,
        "random_state": RANDOM_STATE,
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)

    full_ensemble = 0.5 * _normalize(
        iso_full, norm_stats["iso_lo"], norm_stats["iso_hi"]
    ) + 0.5 * _normalize(ee_full, norm_stats["ee_lo"], norm_stats["ee_hi"])

    report = {
        "n_training_rows": int(X.shape[0]),
        "feature_columns": FEATURE_COLUMNS,
        "threshold": threshold,
        "cross_validation": cv_stats,
        "full_fit_flagged_fraction": float((full_ensemble >= threshold).mean()),
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    print(f"\nSaved detector bundle to {MODEL_PATH}")


if __name__ == "__main__":
    main()
