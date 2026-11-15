"""Run the trained detector on a CSV of HPC traces.

Usage
-----
    python detect.py path/to/traces.csv [--model model/detector.joblib] [--out predictions.csv]

The input CSV must contain the columns: cache-references, cycles, LLC-loads.
If it also contains a "label" column (0 = clean, 1 = backdoor), standard
metrics (Accuracy, TPR, FPR, F1, AUROC) are computed and printed; this is
how the private validation set described in the problem statement can be
scored with this same script.
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

from features import engineer_features

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = ROOT / "model" / "detector.joblib"


def load_bundle(model_path: Path):
    return joblib.load(model_path)


def score_traces(bundle: dict, df: pd.DataFrame) -> np.ndarray:
    feat_df = engineer_features(df)
    X_raw = feat_df[bundle["feature_columns"]].to_numpy()
    X = bundle["scaler"].transform(X_raw)

    iso_score = -bundle["iso_forest"].score_samples(X)
    ee_score = -bundle["elliptic_envelope"].score_samples(X)

    ns = bundle["norm_stats"]
    iso_norm = np.clip(
        (iso_score - ns["iso_lo"]) / (ns["iso_hi"] - ns["iso_lo"] + 1e-12), 0.0, 1.0
    )
    ee_norm = np.clip(
        (ee_score - ns["ee_lo"]) / (ns["ee_hi"] - ns["ee_lo"] + 1e-12), 0.0, 1.0
    )
    return 0.5 * iso_norm + 0.5 * ee_norm


def predict(bundle: dict, df: pd.DataFrame):
    scores = score_traces(bundle, df)
    preds = (scores >= bundle["threshold"]).astype(int)
    return preds, scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=str, help="CSV file of HPC traces to score")
    parser.add_argument(
        "--model", type=str, default=str(DEFAULT_MODEL_PATH), help="Path to detector.joblib"
    )
    parser.add_argument(
        "--out", type=str, default=None, help="Optional path to write predictions CSV"
    )
    args = parser.parse_args()

    bundle = load_bundle(Path(args.model))
    df = pd.read_csv(args.csv_path)

    preds, scores = predict(bundle, df)

    result = df.copy()
    result["anomaly_score"] = scores
    result["prediction"] = preds  # 0 = clean, 1 = backdoor

    if args.out:
        result.to_csv(args.out, index=False)
        print(f"Wrote predictions to {args.out}")

    if "label" in df.columns:
        y_true = df["label"].to_numpy()
        metrics = {
            "accuracy": accuracy_score(y_true, preds),
            "f1": f1_score(y_true, preds, zero_division=0),
            "auroc": roc_auc_score(y_true, scores)
            if len(np.unique(y_true)) > 1
            else float("nan"),
        }
        tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
        metrics["tpr"] = tp / (tp + fn) if (tp + fn) else float("nan")
        metrics["fpr"] = fp / (fp + tn) if (fp + tn) else float("nan")
        print(json.dumps(metrics, indent=2))
    else:
        n_flagged = int(preds.sum())
        print(f"Flagged {n_flagged}/{len(preds)} traces as backdoor "
              f"({100.0 * n_flagged / len(preds):.2f}%)")


if __name__ == "__main__":
    main()
