"""
svd_flu_data.py

flu matrix:
- rows = season_week (0..32)
- columns = one state within one season (i.e., (season, location) pairs)
- values = normalized flu hospitalizations

By applying SVD/PCA, we decompose the messy raw data into three meaningful parts:
-U (Seasonal Shapes): The core "archetypes" of a flu season (e.g., a sharp peak vs. a slow burn). Each vector is 33 weeks long.
-SIGMA (Importance): A ranking of which seasonal shapes are the most common across the entire dataset.
-V (Regional Weights): Tells us which states or seasons align most closely with the shapes found in U

Inputs:
- analysis_data/hosps_pop_norm.csv
- analysis_data/hosps_pop_zscore.csv

Outputs:
- analysis_data/SVD_flu_pop.csv
- analysis_data/SVD_flu_zscore.csv
- analysis_data/SVD_flu_pop_columns.csv
- analysis_data/SVD_flu_zscore_columns.csv
"""

from __future__ import annotations

import os
import sys
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA


POP_PATH = os.path.join("analysis_data", "hosps_pop_norm.csv")
Z_PATH = os.path.join("analysis_data", "hosps_pop_zscore.csv")

OUT_POP = os.path.join("analysis_data", "SVD_flu_pop.csv")
OUT_Z = os.path.join("analysis_data", "SVD_flu_zscore.csv")

OUT_POP_COLMAP = os.path.join("analysis_data", "SVD_flu_pop_columns.csv")
OUT_Z_COLMAP = os.path.join("analysis_data", "SVD_flu_zscore_columns.csv")


def pick_value_column(df: pd.DataFrame, candidates: list[str]) -> str:
    """Pick the first candidate column that exists in df."""
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(
        "Could not find the normalized value column. Tried:\n"
        + "\n".join(f"  - {c}" for c in candidates)
        + f"\nAvailable columns: {list(df.columns)}"
    )


def build_matrix(df: pd.DataFrame, value_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      X_df: rows=season_week, columns=col_id, values=value_col
      colmap: col_id -> season, location, location_name (if present)
    """
    required = {"season_week", "season", "location", value_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Keep only season_week 0..32 and drop missing values
    d = df.copy()
    d = d[(d["season_week"].between(0, 32))].copy()
    d = d[d[value_col].notna()].copy()
    d["location"] = d["location"].astype(str).str.zfill(2)

    # Build a column key (season + location)
    d["col_key"] = d["season"].astype(str) + "__" + d["location"].astype(str)

    # Column mapping (stable ordering)
    keep_cols = ["col_key", "season", "location"]
    if "location_name" in d.columns:
        keep_cols.append("location_name")

    colmap = (
        d[keep_cols]
        .drop_duplicates("col_key")
        .sort_values(["season", "location"])
        .reset_index(drop=True)
    )
    colmap["col_id"] = np.arange(len(colmap))

    # Merge col_id back
    d = d.merge(colmap[["col_key", "col_id"]], on="col_key", how="left")

    # Pivot into matrix: rows=season_week, cols=col_id
    X = d.pivot_table(
        index="season_week",
        columns="col_id",
        values=value_col,
        aggfunc="mean",
    ).sort_index()

    # Ensure all weeks exist (0..32)
    X = X.reindex(range(33))

    # Drop columns with too many missing values (simple rule)
    # If a column is missing ANY week, PCA becomes messy; drop those columns.
    good_cols = X.columns[X.notna().all(axis=0)]
    X = X[good_cols]

    # Keep only mapping rows for columns we kept
    colmap = colmap[colmap["col_id"].isin(good_cols)].copy()
    colmap = colmap.sort_values("col_id").reset_index(drop=True)

    return X, colmap


def pca_svd_to_long(X: pd.DataFrame, n_components: int | None = None) -> pd.DataFrame:
    """
    Compute PCA via SVD on centered data.
    Returns one long dataframe with columns:
      vector_name, vector_number1, season_week, value

    Encoding:
    - U rows: vector_name='u', vector_number1=component, season_week=0..32, value=U[week, comp]
    - SIGMA rows: vector_name='sigma', vector_number1=component, season_week=-1, value=sigma[comp]
    - V rows: vector_name='v', vector_number1=column_id, season_week=component, value=V[column_id, comp]
      (Here 'season_week' stores the component index for V rows, because V is over columns, not weeks.)
    """
    
    X_np = X.to_numpy(dtype=float)
    col_means = np.nanmean(X_np, axis=0)
    X_centered = X_np - col_means

    k = min(X_centered.shape) if n_components is None else min(n_components, min(X_centered.shape))

    pca = PCA(n_components=k, svd_solver="full")
    scores = pca.fit_transform(X_centered)          # = U * SIGMA
    sigma = pca.singular_values_                   # length k
    Vt = pca.components_                           # shape (k, n_cols)
    V = Vt.T                                       # shape (n_cols, k)
    U = scores / sigma                             # shape (n_weeks, k)

    #U long
    u_rows = []
    for comp in range(k):
        for w in range(U.shape[0]):
            u_rows.append(("u", comp, w, float(U[w, comp])))

    #sigma long
    s_rows = [("sigma", comp, -1, float(sigma[comp])) for comp in range(k)]

    #V long
    v_rows = []
    n_cols = V.shape[0]
    for col_idx in range(n_cols):
        for comp in range(k):
            v_rows.append(("v", int(X.columns[col_idx]), comp, float(V[col_idx, comp])))

    out = pd.DataFrame(u_rows + s_rows + v_rows, columns=["vector_name", "vector_number1", "season_week", "value"])
    return out


def run_one(input_path: str, out_path: str, out_colmap_path: str, value_candidates: list[str]) -> None:
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input not found: {input_path}")

    df = pd.read_csv(input_path)

    value_col = pick_value_column(df, value_candidates)
    X, colmap = build_matrix(df, value_col=value_col)

    if X.shape[1] < 2:
        raise ValueError(f"Not enough complete columns for PCA. Columns after filtering: {X.shape[1]}")

    svd_long = pca_svd_to_long(X)

    svd_long.to_csv(out_path, index=False)
    colmap.to_csv(out_colmap_path, index=False)

    print(f"Wrote {out_path}  (rows={len(svd_long):,})")
    print(f"Wrote {out_colmap_path}  (columns kept={X.shape[1]:,})")
    print(f"Matrix shape used for PCA: weeks={X.shape[0]}, columns={X.shape[1]}")
    print(f"Value column used: {value_col}")


def main() -> None:
    # For pop-normalized file, common names could be:
    pop_candidates = ["hosps_pop_norm"]

    # For z-score file, common names could be:
    z_candidates = ["zflu"]

    run_one(POP_PATH, OUT_POP, OUT_POP_COLMAP, pop_candidates)
    run_one(Z_PATH, OUT_Z, OUT_Z_COLMAP, z_candidates)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
