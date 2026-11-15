"""Correlation between fleet daily total distance and daily energy consumption.

Total_Distance.csv has one fleet-wide total per day for 2023. The energy data
carries no year, so the two are joined on month and day. (The weekly pattern
in energy lines up with 2023 weekdays, so 2023 appears to be the right year.)
Energy rows are cleaned as in plot_energy_vs_temperature.py: overlapping
month blocks are de-duplicated and November (two conflicting copies) is dropped.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from plot_energy_vs_temperature import COLORS, load_truck_days, pearson_ci

HERE = Path(__file__).resolve().parent
DISTANCE = HERE.parent / "Other_data" / "Total_Distance.csv"


def month_day(dates: pd.Series) -> pd.Index:
    return pd.Index(dates.dt.strftime("%m-%d"), name="md")


def load_distance() -> pd.Series:
    d = pd.read_csv(DISTANCE, parse_dates=["Date"])
    return pd.Series(d["Total Distance"].to_numpy(), index=month_day(d["Date"]), name="distance")


def daily_total(truck_days: pd.DataFrame, distance: pd.Series) -> pd.DataFrame:
    """Total energy across trucks on each date, joined with fleet distance."""
    e = truck_days.groupby("date")["energy_consumption"].sum()
    e.index = month_day(e.index.to_series())
    return pd.concat([e.rename("energy"), distance], axis=1, join="inner")


def main():
    truck_days = load_truck_days()
    distance = load_distance()
    fleet = daily_total(truck_days, distance)

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(15, 6), dpi=180, gridspec_kw={"width_ratios": [3, 2]})

    months = pd.to_numeric(fleet.index.str[:2])
    x, y = fleet["distance"] / 1000, fleet["energy"]
    sc = ax.scatter(x, y, c=months, cmap="viridis", s=22, alpha=0.85)
    fit = stats.linregress(x, y)
    xs = np.linspace(x.min(), x.max(), 100)
    n, xm = len(fleet), x.mean()
    resid_se = np.sqrt(np.sum((y - (fit.intercept + fit.slope * x)) ** 2) / (n - 2))
    band = 1.96 * resid_se * np.sqrt(1 / n + (xs - xm) ** 2 / np.sum((x - xm) ** 2))
    yfit = fit.intercept + fit.slope * xs
    ax.fill_between(xs, yfit - band, yfit + band, color="#b8434f", alpha=0.18, lw=0)
    ax.plot(xs, yfit, color="#b8434f", lw=2)
    r, p, lo, hi = pearson_ci(x, y)
    ax.text(0.02, 0.97,
            f"Pearson r = {r:.2f} (95% CI {lo:.2f} to {hi:.2f}), p = {p:.2g}\n"
            f"slope = {fit.slope:.1f} per 1000 km, n = {n} days",
            transform=ax.transAxes, va="top", fontsize=10,
            bbox=dict(boxstyle="round", fc="white", ec="#cccccc"))
    cb = fig.colorbar(sc, ax=ax, pad=0.01)
    cb.set_label("Month")
    ax.set_xlabel("Fleet total distance per day (1000 km)")
    ax.set_ylabel("Fleet total energy per day (dataset units)")
    ax.set_title("Daily energy vs daily distance — all trucks (Jan–Oct)", fontsize=13)
    ax.grid(color="#e5e5e5")
    ax.set_axisbelow(True)

    rows = []
    for cat, g in truck_days.groupby("truck_category"):
        d = daily_total(g, distance)
        rows.append((cat, *pearson_ci(d["distance"], d["energy"])))
    rc = pd.DataFrame(rows, columns=["cat", "r", "p", "lo", "hi"]).sort_values("r")
    yy = np.arange(len(rc))
    bx.barh(yy, rc["r"], color=[COLORS[c] for c in rc["cat"]], height=0.6)
    bx.errorbar(rc["r"], yy, xerr=[rc["r"] - rc["lo"], rc["hi"] - rc["r"]], fmt="none", ecolor="black", capsize=4)
    bx.axvline(0, color="black", lw=0.8)
    bx.set_yticks(yy, rc["cat"])
    bx.set_xlim(-1, 1)
    bx.set_xlabel("Pearson r (fleet daily distance vs category daily energy)")
    bx.set_title("Correlation by truck category (95% CI)", fontsize=12)
    bx.grid(axis="x", color="#e5e5e5")
    bx.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(HERE / "daily_energy_vs_distance.png")
    plt.close(fig)

    print(f"fleet: r={r:.3f} [{lo:.3f}, {hi:.3f}] p={p:.3g} slope={fit.slope:.2f}/1000km n={n}")
    print(rc.round(3).to_string(index=False))
    print("lowest-distance days:\n", fleet.nsmallest(3, "distance").round(1))


if __name__ == "__main__":
    main()
