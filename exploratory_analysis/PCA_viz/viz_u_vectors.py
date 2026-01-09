"""
viz_u_vectors.py

Visualize the first several principal component time-series vectors (U vectors)
from the saved SVD/PCA outputs.

Inputs:
- analysis_data/SVD_flu_pop.csv
- analysis_data/SVD_flu_zscore.csv

Outputs:
- exploratory_analysis/PCA_viz/pop_norm_viz.pdf
- exploratory_analysis/PCA_viz/z_norm_viz.pdf

Notes:
- We plot the first 3 U vectors (components 0, 1, 2) vs season_week (0..32).
- y-axis is left as a generic "y" as requested.
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


POP_IN = os.path.join("analysis_data", "SVD_flu_pop.csv")
Z_IN = os.path.join("analysis_data", "SVD_flu_zscore.csv")

OUT_DIR = os.path.join("exploratory_analysis", "PCA_viz")
POP_OUT = os.path.join(OUT_DIR, "pop_norm_viz.pdf")
Z_OUT = os.path.join(OUT_DIR, "z_norm_viz.pdf")


def load_u_vectors(path: str, n_components: int = 3) -> pd.DataFrame:
    """Return dataframe with rows for U vectors only, filtered to first n components."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing input: {path}")

    df = pd.read_csv(path)

    required = {"vector_name", "vector_number1", "season_week", "value"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")

    u = df[df["vector_name"].str.lower() == "u"].copy()
    u = u[u["vector_number1"].between(0, n_components - 1)]
    u = u[u["season_week"].between(0, 32)].copy()

    u["vector_number1"] = u["vector_number1"].astype(int)
    u["season_week"] = u["season_week"].astype(int)

    return u


def plot_u_pdf(u: pd.DataFrame, out_pdf: str, title_prefix: str) -> None:
    os.makedirs(os.path.dirname(out_pdf), exist_ok=True)

    # All U vectors on one plot
    with PdfPages(out_pdf) as pdf:
        plt.figure(figsize=(10, 5))
        for k in sorted(u["vector_number1"].unique()):
            uk = u[u["vector_number1"] == k].sort_values("season_week")
            plt.plot(uk["season_week"], uk["value"], label=f"U{k+1}")

        plt.xlabel("season_week")
        plt.ylabel("y")
        plt.title(f"{title_prefix}: First U vectors (overlay)")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        #One component per page
        for k in sorted(u["vector_number1"].unique()):
            uk = u[u["vector_number1"] == k].sort_values("season_week")

            plt.figure(figsize=(10, 4))
            plt.plot(uk["season_week"], uk["value"])
            plt.xlabel("season_week")
            plt.ylabel("y")
            plt.title(f"{title_prefix}: U{k+1}")
            plt.grid(alpha=0.3)
            plt.tight_layout()
            pdf.savefig()
            plt.close()


def main():
    u_pop = load_u_vectors(POP_IN, n_components=3)
    plot_u_pdf(u_pop, POP_OUT, "Population-normalized PCA")

    u_z = load_u_vectors(Z_IN, n_components=3)
    plot_u_pdf(u_z, Z_OUT, "Z-score normalized PCA")

    print(f"Wrote {POP_OUT}")
    print(f"Wrote {Z_OUT}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
