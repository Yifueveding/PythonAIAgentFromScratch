"""Shared loader for all_trucks_energy_consumption_data_with_category.csv.

The raw file has no calendar date. ``day`` is days since the 1st of ``month``
and each month block runs 34 days, so it overlaps the next month by 3-4 days
with identical rows. November holds two conflicting copies of every slot and
cannot be reconciled, so dates from 1 November onwards are dropped.

The file has no year either. Its weekly pattern matches 2023 weekdays (and the
2023 fleet distance file), so 2023 is the default. Pass another non-leap year
only to align with an external dataset on month and day.
"""
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ENERGY = HERE / "all_trucks_energy_consumption_data_with_category.csv"


def load_energy(year: int = 2023) -> pd.DataFrame:
    """Half-hourly rows with a calendar ``date`` column, de-duplicated, Jan-Oct only."""
    df = pd.read_csv(ENERGY, dtype={"month": str})
    df["date"] = pd.to_datetime(f"{year}-" + df["month"] + "-01") + pd.to_timedelta(df["day"], unit="D")
    df = df[df["date"] < pd.Timestamp(f"{year}-11-01")]
    return df.drop_duplicates(["TruckId", "date", "hour", "energy_consumption"]).drop(columns=["month", "day"])
