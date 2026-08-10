"""
Build feature matrices from the MIMIC-III-Ext-PPG metadata file.

Two feature sets:
  V1 (13 features) - the per-segment summary values the dataset ships with,
                     plus patient demographics.
  V2 (26 features) - V1 plus 13 features engineered from the three 10-second
                     windows inside each 30-second segment, capturing how much
                     heart rate and blood pressure move within the segment.

Note on what these features are: they come from metadata.csv, not from the
PPG waveform. Heart rate in that file is derived from the concurrent ECG
R-R intervals and blood pressure from the arterial line. See the data note
in the README.

Usage:
    python src/build_features.py --data /path/to/mimic-iii-ext-ppg/1.1.0 \
                                 --out features.npz
"""

import argparse
import re

import numpy as np
import pandas as pd

SCALAR_COLS = [
    "median_30s_hr", "iqr_30s_hr",
    "median_30s_rr", "iqr_30s_rr",
    "median_30s_sbp", "iqr_30s_sbp",
    "median_30s_dbp", "iqr_30s_dbp",
    "resp_sqi", "age", "gender", "weight", "height",
]
VECTOR_COLS = ["vector_10s_hr", "vector_10s_median_sbp", "vector_10s_median_dbp"]

# the four features that are constant within a patient
DEMOGRAPHIC_COLS = ["age", "gender", "weight", "height"]


def parse_window_vector(val):
    """Parse a stringified 3-element array into floats, padding with NaN."""
    out = np.full(3, np.nan, dtype=np.float32)
    if not isinstance(val, str):
        return out
    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", val)
    if nums:
        arr = np.array([float(x) for x in nums[:3]], dtype=np.float32)
        out[: len(arr)] = arr
    return out


def build(data_dir):
    cols = SCALAR_COLS + VECTOR_COLS + ["event_rhythm", "patient"]
    meta = pd.read_csv(f"{data_dir}/metadata.csv", usecols=cols)
    print(f"Loaded {len(meta):,} segments from {meta['patient'].nunique():,} patients")

    y = (meta["event_rhythm"] == "AF").values.astype(np.float32)
    patient_id = meta["patient"].values
    meta["gender"] = (meta["gender"] == "M").astype(float)

    v1 = meta[SCALAR_COLS].fillna(meta[SCALAR_COLS].median()).values.astype(np.float32)

    hr = np.stack(meta["vector_10s_hr"].apply(parse_window_vector).values)
    sbp = np.stack(meta["vector_10s_median_sbp"].apply(parse_window_vector).values)
    dbp = np.stack(meta["vector_10s_median_dbp"].apply(parse_window_vector).values)

    # Segments with no BP or respiratory data at all produce all-NaN slices here.
    # They are zero-filled below, which means "missing" and "zero" are
    # indistinguishable to the model. This is a known limitation.
    with np.errstate(all="ignore"):
        hr_mean, hr_std = np.nanmean(hr, 1), np.nanstd(hr, 1)
        sbp_mean, dbp_mean = np.nanmean(sbp, 1), np.nanmean(dbp, 1)
        d01 = np.abs(hr[:, 1] - hr[:, 0])
        d12 = np.abs(hr[:, 2] - hr[:, 1])

        v2 = np.column_stack([
            hr_mean,                                   # mean HR across windows
            hr_std,                                    # HR spread
            np.nanmax(hr, 1) - np.nanmin(hr, 1),       # HR range
            d01,                                       # window 0 -> 1 change
            d12,                                       # window 1 -> 2 change
            hr_std / (hr_mean + 1e-8),                 # coefficient of variation
            d01 + d12,                                 # total HR irregularity
            sbp_mean,
            np.nanstd(sbp, 1),
            np.nanmax(sbp, 1) - np.nanmin(sbp, 1),
            dbp_mean,
            np.nanstd(dbp, 1),
            sbp_mean - dbp_mean,                       # pulse pressure
            v1,
        ]).astype(np.float32)

    v2 = np.nan_to_num(v2, nan=0.0, posinf=0.0, neginf=0.0)

    n_missing = int(np.isnan(hr).all(axis=1).sum())
    print(f"V1 {v1.shape}   V2 {v2.shape}")
    print(f"AF prevalence {y.mean():.4f}")
    print(f"Segments with no heart rate data at all: {n_missing:,}")
    return v1, v2, y, patient_id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True,
                    help="directory containing metadata.csv")
    ap.add_argument("--out", default="features.npz")
    args = ap.parse_args()

    v1, v2, y, pid = build(args.data)
    np.savez_compressed(args.out, v1=v1, v2=v2, y=y, patient_id=pid)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
