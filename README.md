# data-driven-svd-regression
Dependencies:
- pandas
- epiweeks

Install with:
pip install epiweeks

Data Cleaning and Missing Values

The influenza hospitalization dataset
(analysis_data/target-hospital-admissions.csv) 
contains a subset of rows with missing values, particularly in the following fields:
-location (state identifier)
-location_name
-value (number of hospital admissions)
-weekly_rate
These missing values originate from the upstream CDC/Socrata data source and reflect incomplete or non–state-level records rather than data processing errors.

Handling strategy
We do not impute missing values in this dataset.
-Rows with missing location or value are excluded prior to population merging and seasonal feature construction, as they cannot be reliably interpreted or linked to state-level population data.
-Missing values in weekly_rate are not imputed, since this field is derived and can be recomputed after population data are merged.
This approach prioritizes data integrity and aligns with standard epidemiological data-cleaning practices.