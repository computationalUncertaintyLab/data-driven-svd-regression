"""
format_flu_data.py
- Add MMWR epiweek variables + flu season variables to weekly flu admissions data.
- Merge state population from locations file.

This will Generate:
- analysis_data/formatted_flu.csv

Season definition:
- Flu season starts at MMWR week 40.
- season is labeled as "YYYY/YYYY+1" where YYYY is the season start year.
- season_week is 0 at the start of week 40 and ends at 32 (33 weeks total: 0..32).
"""

from __future__ import annotations

import os
import sys
import pandas as pd
from epiweeks import Week

#files for INput & Output
ADM_PATH = os.path.join("data", "target-hospital-admissions_raw.csv")
OUT_PATH = os.path.join("analysis_data", "formatted_flu.csv")

# We'll try these in order:
LOC_CANDIDATES = [
    os.path.join("data", "locations.csv"),
    os.path.join("data", "from_state_to_fip_and_pop.csv"),
]


def _pick_locations_file() -> str:
    """Look through our list of possible location files and use the first one that exists"""
    for p in LOC_CANDIDATES:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        "Could not find a locations file. Expected one of:\n"
        + "\n".join(f"  - {p}" for p in LOC_CANDIDATES)
    )


def normalize_location(x) -> str:
    """
    Cleaning up location codes so they match between different files.
    Examples:1 or "1" or "1.0" -> "01"
      "US" stays "US"
    """
    s = str(x).strip()
    if s.endswith(".0"):
        s = s[:-2]
    if s.upper() == "US":
        return "US"
    if s.isdigit():
        return s.zfill(2)
    return s


def _mmwr_info(dt: pd.Timestamp) -> tuple[int, int]:
    """Figure out which epidemiological year and week a date falls into"""
    w = Week.fromdate(dt.to_pydatetime().date())
    return w.year, w.week


def _season_start_year(epiyear: int, epiweek: int) -> int:
    """
    If epiweek >= 40 -> season starts in epiyear.
    Else (weeks 1..39) -> season started previous epiyear.
    """
    return epiyear if epiweek >= 40 else (epiyear - 1)


def main() -> None:
    # Load flu admissions data
    #Checking if the admissions file exists before trying to open it
    if not os.path.exists(ADM_PATH):
        raise FileNotFoundError(f"Admissions input not found: {ADM_PATH}")

    adm = pd.read_csv(ADM_PATH)

    # Check required columns first
    required_cols = {"date", "location", "location_name", "value"}
    missing = required_cols - set(adm.columns)
    if missing:
        raise ValueError(f"Admissions file missing required columns: {missing}")

    # Drop rows with missing critical fields
    adm = adm[adm["location"].notna()].copy()
    adm = adm[adm["value"].notna()].copy()

    # Normalize location codes before any merge
    adm["location"] = adm["location"].apply(normalize_location)

    # Parsing the dates
    adm["date"] = pd.to_datetime(adm["date"], errors="coerce")
    if adm["date"].isna().any():
        bad_rows = adm[adm["date"].isna()].head(5)
        raise ValueError(
            "Some 'date' values could not be parsed. Example rows:\n"
            + bad_rows.to_string(index=False)
        )

    # Computing epiweek/epiyear
    # For each date, figure out its epi-year and epi-week (CDC's special calendar system)
    epi = adm["date"].apply(_mmwr_info)
    adm["epiyear"] = epi.apply(lambda x: x[0])
    adm["epiweek"] = epi.apply(lambda x: x[1])

    #Computing season + season_week
    adm["season_start_year"] = adm.apply(
        lambda r: _season_start_year(int(r["epiyear"]), int(r["epiweek"])),
        axis=1,
    )
    # Creating season labels like "2023/2024" for easy reading
    adm["season"] = adm["season_start_year"].apply(lambda y: f"{y}/{y+1}")

    # Finding the actual start date of each flu season (week 40 of the start year)
    season_start_date = {
        y: Week(y, 40).startdate() for y in adm["season_start_year"].unique()
    }

    # season_week = whole weeks since season start (0 = first week, 1 = second week, etc.)
    adm["season_week"] = adm.apply(
        lambda r: int(
            (r["date"].date() - season_start_date[int(r["season_start_year"])]).days // 7
        ),
        axis=1,
    )

    # Keep only the defined season window (0..32)
    adm = adm[(adm["season_week"] >= 0) & (adm["season_week"] <= 32)].copy()

    # Load locations + merge population
    loc_path = _pick_locations_file()
    loc = pd.read_csv(loc_path)

    if "location" not in loc.columns or "population" not in loc.columns:
        raise ValueError(
            f"Locations file {loc_path} must contain columns: 'location' and 'population'."
        )

    loc_small = loc[["location", "population"]].copy()
    loc_small["location"] = loc_small["location"].apply(normalize_location)

    # Combining the admissions data with the population data based on location codes
    out = adm.merge(loc_small, on="location", how="left")

    # Population must exist for every row after merge
    if out["population"].isna().any():
        missing_locs = sorted(out.loc[out["population"].isna(), "location"].unique().tolist())
        raise ValueError(
            "Population missing for some locations after merge. "
            f"Examples: {missing_locs[:10]}"
        )

    # Final column order and saving the file
    preferred_order = [
        "date",
        "location",
        "location_name",
        "population",
        "value",
        "season",
        "season_week",
        "epiyear",
        "epiweek",
    ]
    cols = [c for c in preferred_order if c in out.columns] + [
        c for c in out.columns if c not in preferred_order and c != "season_start_year"
    ]
    
    #Sort the data by season, then location, then week for logical ordering
    out = out[cols].sort_values(["season", "location", "season_week", "date"])

    out.to_csv(OUT_PATH, index=False)
    print(f"Wrote {OUT_PATH} with {len(out):,} rows.")
    print(f"Locations file used: {loc_path}")
    print(f"season_week range: {out['season_week'].min()}..{out['season_week'].max()}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
