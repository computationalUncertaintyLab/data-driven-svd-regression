# data-driven-svd-regression
Dependencies:
- pandas
- epiweeks

Install with:
pip install epiweeks

# Project structure

-data/
Raw reference data (not modified).
    -from_state_to_fip_and_pop.csv
    -get_target_data.R

-analysis_data/
 Generated datasets and scripts.
    -target-hospital-admissions.csv
    -format_flu_data.py
    -pop_norm_flu.py
    -z_score_flu.py

-exploratory_analysis/
 Jupyter notebooks for validation plots and exploration

# Data Cleaning and Missing Values

The influenza hospitalization dataset
(analysis_data/target-hospital-admissions.csv) 
contains a subset of rows with missing values, particularly in the following fields:
-location (state identifier)
-location_name
-value (number of hospital admissions)
-weekly_rate
These missing values originate from the upstream CDC/Socrata data source and reflect incomplete or non–state-level records rather than data processing errors.

# Handling strategy
We do not impute missing values in this dataset.
-Rows with missing location or value are excluded prior to population merging and seasonal feature construction, as they cannot be reliably interpreted or linked to state-level population data.
-Missing values in weekly_rate are not imputed, since this field is derived and can be recomputed after population data are merged.
This approach prioritizes data integrity and aligns with standard epidemiological data-cleaning practices.

# Z score normalization
Z-score normalization centers each state's hospitalization time series at zero
and scales by its historical standard deviation, enabling comparison of
relative flu intensity across states with very different population sizes.

# PCA
We constructed a flu matrix with season weeks as rows and state-season pairs as columns, then applied PCA via SVD to both population-normalized and z-score–normalized datasets. We stored the resulting U, Σ, and V matrices in long-form CSVs for full reproducibility
