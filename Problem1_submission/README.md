# HPC-Based Backdoor Detection (Problem 1)

Anomaly detector that flags backdoor-triggered inferences from hardware
performance counter (HPC) traces, trained **only on clean/benign traces**
(strict anomaly-detection setting — no backdoor examples are available at
training time).

## Data

`data/trace.csv` — 800 rows of benign-inference HPC traces, 3 raw counters:

- `cache-references`
- `cycles`
- `LLC-loads`

## Method

See [`report.pdf`](report.pdf) for the full write-up. In short:

1. **Feature engineering** (`src/features.py`): the 3 raw counters plus 3
   ratio features (`cacheref_per_cycle`, `llcload_per_cycle`,
   `llcload_per_cacheref`). Ratios expose shifts in the *relationship*
   between counters, which is more robust to a backdoor trigger than any
   single counter's absolute value.
2. **Scaling**: `RobustScaler` (median/IQR), robust to the mild skew in
   `cycles`.
3. **Detector**: an ensemble of two one-class models trained on clean
   traces only —
   - `IsolationForest` (non-linear / local density structure)
   - `EllipticEnvelope` (robust Mahalanobis distance / global covariance)

   Each model's anomaly score is min-max normalized and the two are
   averaged into a single ensemble score. Averaging two models with
   different inductive biases reduces the risk of the whole detector
   being blind to a type of shift that just one method would miss.
4. **Threshold selection**: 5-fold cross-validation on the clean data —
   in each fold, models are fit on the training rows and scored on the
   held-out rows; the pooled held-out scores (all still clean) set the
   threshold at the 98th percentile, i.e. calibrated for a ~2% false
   positive rate on unseen clean traces. The final shipped models are
   refit on the full 800-row clean set; the cross-validated threshold is
   kept since it already reflects held-out generalization.

Trained bundle (scaler + both models + normalization stats + threshold)
is saved to `model/detector.joblib`.

## Repository layout

```
data/trace.csv          # provided benign traces
src/features.py         # feature engineering
src/train.py            # trains + cross-validates + saves the detector
src/detect.py           # inference / evaluation CLI
model/detector.joblib   # saved detector bundle (scaler, models, threshold)
model/training_report.json  # metrics from the last training run
tests/sanity_check.py   # builds a synthetic proxy anomaly set (NOT real
                         # backdoor data) to sanity-check the pipeline end-to-end
report.pdf              # short report describing the method
```

## Usage

```bash
pip install -r requirements.txt

# retrain (reproduces model/detector.joblib from data/trace.csv)
python src/train.py

# score a CSV of traces (columns: cache-references, cycles, LLC-loads[, label])
python src/detect.py path/to/traces.csv --out predictions.csv
```

If the input CSV has a `label` column (0 = clean, 1 = backdoor), `detect.py`
prints Accuracy, F1, AUROC, TPR and FPR directly — this is the same script
intended for scoring on the private validation set.

### Sanity check (synthetic, not real backdoor data)

```bash
python tests/sanity_check.py
python src/detect.py tests/_synthetic_eval.csv
```

This perturbs real clean traces (more LLC pressure, fewer cycles for the
same work) to build a toy proxy anomaly set, purely to confirm the
detector reacts to distributional shifts before it is evaluated on the
real private set.

## Output

`detect.py` adds two columns to the input traces:

- `anomaly_score` — ensemble score in `[0, 1]`, higher = more anomalous
- `prediction` — `0` (clean) or `1` (backdoor), thresholded at the
  cross-validated cutoff stored in the model bundle
