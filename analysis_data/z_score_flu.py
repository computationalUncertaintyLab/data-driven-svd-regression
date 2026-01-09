"""
z_score_flu.py

We convert flu hospitalization numbers into z-scores, which tell us how unusual
the current flu activity is compared to what's typical for each state.

The math:
zflu = (hospitalizations - state_mean) / state_std

Input file needed:
- analysis_data/target-hospital-admissions.csv
- data/from_state_to_fip_and_pop.csv

Output file created:
- analysis_data/hosps_pop_zscore.csv
"""

import os
import sys
import pandas as pd

#File paths for Input & Output
ADM_PATH = os.path.join("analysis_data", "formatted_flu.csv")
OUT_PATH = os.path.join("analysis_data", "hosps_pop_zscore.csv")


def main():
    
    # Loading data
    if not os.path.exists(ADM_PATH):
        raise FileNotFoundError(f"Missing input file: {ADM_PATH}")

    df = pd.read_csv(ADM_PATH)
    df["date"] = pd.to_datetime(df["date"])

    # Keeping valid rows only
    df = df[df["location"].notna()]
    df = df[df["value"].notna()].copy()

    # Ensuring location is of string datatype
    df["location"] = df["location"].astype(str)

    # Computing the z-score per state
    stats = (
        df.groupby("location")["value"]
        .agg(["mean", "std"])
        .rename(columns={"mean": "state_mean", "std": "state_std"})
        .reset_index()
    )

    df = df.merge(stats, on="location", how="left")

    #Remove any states where the standard deviation is zero 
    #Avoiding division by zero
    df = df[df["state_std"] > 0].copy()

    #z-Score
    df["zflu"] = (df["value"] - df["state_mean"]) / df["state_std"]

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
