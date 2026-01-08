"""
z_score_flu.py

Goal:
- Convert flu hospitalizations to z-scores per state.

Definition:
    zflu = (hospitalizations - state_mean) / state_std

Inputs:
- analysis_data/target-hospital-admissions.csv
- data/from_state_to_fip_and_pop.csv (for population merge consistency)

Output:
- analysis_data/hosps_pop_zscore.csv
"""

import os
import sys
import pandas as pd

ADM_PATH = os.path.join("analysis_data", "target-hospital-admissions.csv")
OUT_PATH = os.path.join("analysis_data", "hosps_pop_zscore.csv")


def main():
    # -----------------------------
    # Load data
    # -----------------------------
    if not os.path.exists(ADM_PATH):
        raise FileNotFoundError(f"Missing input file: {ADM_PATH}")

    df = pd.read_csv(ADM_PATH)
    df["date"] = pd.to_datetime(df["date"])

    # Keep valid rows
    df = df[df["location"].notna()]
    df = df[df["value"].notna()].copy()

    # Ensure location is string (important for state codes)
    df["location"] = df["location"].astype(str)

    # -----------------------------
    # Compute z-score per state
    # -----------------------------
    stats = (
        df.groupby("location")["value"]
        .agg(["mean", "std"])
        .rename(columns={"mean": "state_mean", "std": "state_std"})
        .reset_index()
    )

    df = df.merge(stats, on="location", how="left")

    # Avoid division by zero
    df = df[df["state_std"] > 0].copy()

    df["zflu"] = (df["value"] - df["state_mean"]) / df["state_std"]

    # -----------------------------
    # Save output
    # -----------------------------
    df.to_csv(OUT_PATH, index=False)

    print(f"Wrote {OUT_PATH}")
    print(f"States: {df['location'].nunique()}")
    print(f"zflu range: {df['zflu'].min():.2f} to {df['zflu'].max():.2f}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
