"""Hourly average energy by time of day (all trucks, and per truck category).

The raw data has 48 half-hour slots per day (column ``hour`` = 0..47).
Two consecutive slots are summed per truck/month/day to get hourly energy,
then averaged across all truck-hours, with a 95% confidence interval
(mean +/- 1.96 * standard error).
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "all_trucks_energy_consumption_data_with_category.csv"

COLORS = {
    "Dril_Blast": "#2a6ebb", "Fixed_plant": "#e08a1e", "Ghosts": "#3a9a6a",
    "Mobile_Eq": "#b8434f", "Production": "#7e5fa6", "Rail": "#3fa3a3",
    "Support": "#ab7a2e", "Tech_services": "#6b7280",
}


def load_hourly() -> pd.DataFrame:
    """One row per (truck, active day, hour of day) with hourly energy.

    The CSV only contains slots where energy was recorded, so a missing
    half-hour slot means zero consumption. Every active truck-day gets all
    24 hours, with absent hours filled with 0, so the average is not biased
    by how many slots happen to be present at each time of day.
    """
    df = pd.read_csv(DATA, dtype={"month": str})
    df["hour_of_day"] = df["hour"] // 2
    day_keys = ["TruckId", "truck_category", "month", "day"]
    wide = (
        df.groupby(day_keys + ["hour_of_day"])["energy_consumption"].sum()
        .unstack("hour_of_day")
        .reindex(columns=range(24))
        .fillna(0.0)
    )
    return wide.stack().rename("energy_consumption").reset_index()


def summarize(hourly: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    g = hourly.groupby(by)["energy_consumption"].agg(["mean", "std", "count"]).reset_index()
    half = 1.96 * g["std"] / np.sqrt(g["count"])
    g["lo"], g["hi"] = g["mean"] - half, g["mean"] + half
    return g


def style(ax, title):
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Mean energy consumption per hour (dataset units)")
    ax.set_xticks(range(24))
    ax.set_xlim(0, 23)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", color="#e5e5e5")
    ax.set_axisbelow(True)


def main():
    hourly = load_hourly()

    s = summarize(hourly, ["hour_of_day"])
    fig, ax = plt.subplots(figsize=(12, 6), dpi=180)
    ax.fill_between(s["hour_of_day"], s["lo"], s["hi"], color="#2a6ebb", alpha=0.2, lw=0, label="95% CI")
    ax.plot(s["hour_of_day"], s["mean"], color="#2a6ebb", lw=2, marker="o", ms=4, label="All trucks")
    style(ax, "Average hourly energy by time of day — all trucks")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(HERE / "all_trucks_average_energy_by_time_of_day_ci.png")
    plt.close(fig)

    s = summarize(hourly, ["truck_category", "hour_of_day"])
    fig, ax = plt.subplots(figsize=(12, 6), dpi=180)
    for cat, d in s.groupby("truck_category"):
        c = COLORS.get(cat)
        ax.fill_between(d["hour_of_day"], d["lo"], d["hi"], color=c, alpha=0.15, lw=0)
        ax.plot(d["hour_of_day"], d["mean"], color=c, lw=2, marker="o", ms=3, label=cat)
    style(ax, "Average hourly energy by time of day — by truck category (shaded: 95% CI)")
    ax.legend(frameon=False, loc="upper left", ncol=2)
    fig.tight_layout()
    fig.savefig(HERE / "truck_category_average_energy_by_time_of_day_ci.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
