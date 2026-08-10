"""
Compare patient-level and segment-level data splitting on the same data,
the same features, and the same model.

The only thing that differs between the two arms is where the train/test
boundary falls: between patients, or between segments. Everything else is
held fixed, so the difference in reported performance is the inflation
caused by splitting at the wrong level.

A third feature set, demographics only (age, sex, weight, height), is
included as a control. Those four values are constant within a patient and
carry no information about which rhythm the patient is in at a given moment,
so any performance above the population base rate has to come from the model
recognizing individual patients.

Usage:
    python src/leakage_experiment.py --features features.npz \
                                     --subsample 1000000 \
                                     --out results/results.json
"""

import argparse
import gc
import json

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.utils.class_weight import compute_sample_weight

SEED = 42
HOLDOUT_FOLD = 4
N_FOLDS = 5


def patient_level_folds(patient_id, y, seed=SEED):
    """Assign folds to patients. Every segment from a patient lands in one fold."""
    patients = np.unique(patient_id)
    has_af = np.array([y[patient_id == p].mean() > 0 for p in patients]).astype(int)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    assign = np.full(len(patients), -1)
    for fold, (_, val_idx) in enumerate(skf.split(patients, has_af)):
        assign[val_idx] = fold
    lookup = dict(zip(patients, assign))
    return np.array([lookup[p] for p in patient_id])


def segment_level_folds(y, seed=SEED):
    """Assign folds to segments directly. Patients end up on both sides."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    assign = np.full(len(y), -1)
    for fold, (_, val_idx) in enumerate(skf.split(np.zeros(len(y)), y)):
        assign[val_idx] = fold
    return assign


def evaluate(X, y, patient_id, folds, label):
    dev = folds != HOLDOUT_FOLD
    test = folds == HOLDOUT_FOLD

    # Class weighting rather than resampling: same correction for the 9.4%
    # AF prevalence without duplicating the training set in memory.
    weights = compute_sample_weight("balanced", y[dev])

    model = HistGradientBoostingClassifier(
        max_iter=300, max_depth=5, learning_rate=0.1, random_state=SEED,
    )
    model.fit(X[dev], y[dev], sample_weight=weights)

    test_probs = model.predict_proba(X[test])[:, 1]
    dev_probs = model.predict_proba(X[dev])[:, 1]

    # Threshold swept on the development set only, identical procedure for
    # both arms so the comparison stays fair.
    grid = np.arange(0.05, 0.96, 0.01)
    scores = [f1_score(y[dev], (dev_probs >= t).astype(int), zero_division=0)
              for t in grid]
    threshold = float(grid[int(np.argmax(scores))])

    preds = (test_probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y[test], preds).ravel()

    result = {
        "label": label,
        "auroc": float(roc_auc_score(y[test], test_probs)),
        "average_precision": float(average_precision_score(y[test], test_probs)),
        "f1": float(f1_score(y[test], preds, zero_division=0)),
        "precision": float(precision_score(y[test], preds, zero_division=0)),
        "recall": float(recall_score(y[test], preds, zero_division=0)),
        "specificity": float(tn / (tn + fp)),
        "threshold": threshold,
        "n_test_segments": int(test.sum()),
        "n_test_patients": int(len(np.unique(patient_id[test]))),
        "patients_in_both_train_and_test": int(
            len(set(patient_id[dev]) & set(patient_id[test]))
        ),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp),
                             "fn": int(fn), "tp": int(tp)},
    }

    print(f"  {label:34s} AUROC={result['auroc']:.4f}  "
          f"AP={result['average_precision']:.4f}  F1={result['f1']:.4f}")
    print(f"  {'':34s} patients in both train and test: "
          f"{result['patients_in_both_train_and_test']:,}")

    del model, test_probs, dev_probs, weights
    gc.collect()
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="features.npz")
    ap.add_argument("--subsample", type=int, default=1_000_000,
                    help="0 for the full dataset")
    ap.add_argument("--out", default="results/results.json")
    args = ap.parse_args()

    data = np.load(args.features, allow_pickle=True)
    v1, v2, y, patient_id = data["v1"], data["v2"], data["y"], data["patient_id"]

    idx = np.arange(len(y))
    if args.subsample:
        idx = np.random.RandomState(SEED).choice(
            idx, min(args.subsample, len(idx)), replace=False)

    y_s, pid_s = y[idx], patient_id[idx]
    print(f"{len(idx):,} segments   {len(np.unique(pid_s)):,} patients   "
          f"AF rate {y_s.mean():.4f}")

    folds_patient = patient_level_folds(pid_s, y_s)
    folds_segment = segment_level_folds(y_s)

    feature_sets = [
        ("V1 aggregate", v1[idx]),
        ("V2 + irregularity", v2[idx]),
        ("demographics only", v1[idx][:, -4:]),
    ]

    results = []
    for name, X in feature_sets:
        print(f"\n{name}")
        results.append(evaluate(X, y_s, pid_s, folds_patient, f"{name} | patient"))
        results.append(evaluate(X, y_s, pid_s, folds_segment, f"{name} | segment"))
        del X
        gc.collect()

    payload = {
        "n_segments": int(len(idx)),
        "n_patients": int(len(np.unique(pid_s))),
        "af_prevalence": float(y_s.mean()),
        "subsample_seed": SEED,
        "results": results,
    }
    with open(args.out, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
