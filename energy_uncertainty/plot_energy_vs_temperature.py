"""Correlation between daily average temperature and daily energy consumption.

Energy data: ``day`` is days since the 1st of ``month`` (each month block runs
34 days and overlaps the next month by a few days with identical rows), so the
calendar date is ``2019-<month>-01 + day``. Overlapping copies are dropped.
November holds two conflicting copies of every slot, so dates from
2019-11-01 onwards are excluded.

Temperature: hourly MERRA-2 2 m temperature (Renewables.ninja), averaged per
local (Australia/Perth) calendar day.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
ENERGY = HERE / "all_trucks_energy_consumption_data_with_category.csv"
TEMPERATURE = HERE.parent / "Other_data" / "temperature.csv"
CUTOFF = pd.Timestamp("2019-11-01")  # November data is internally inconsistent

COLORS = {
    "Dril_Blast": "#2a6ebb", "Fixed_plant": "#e08a1e", "Ghosts": "#3a9a6a",
    "Mobile_Eq": "#b8434f", "Production": "#7e5fa6", "Rail": "#3fa3a3",
    "Support": "#ab7a2e", "Tech_services": "#6b7280",
}


def load_truck_days() -> pd.DataFrame:
    """Daily energy per truck per calendar date."""
    df = pd.read_csv(ENERGY, dtype={"month": str})
    df["date"] = pd.to_datetime("2019-" + df["month"] + "-01") + pd.to_timedelta(df["day"], unit="D")
    df = df[df["date"] < CUTOFF]
    df = df.drop_duplicates(["TruckId", "date", "hour", "energy_consumption"])
    return df.groupby(["TruckId", "truck_category", "date"], as_index=False)["energy_consumption"].sum()


def load_daily_temperature() -> pd.Series:
    t = pd.read_csv(TEMPERATURE, comment="#", parse_dates=["local_time"])
    return t.groupby(t["local_time"].dt.normalize())["t2m"].mean().rename("temp")


def daily_mean(truck_days: pd.DataFrame, temp: pd.Series) -> pd.DataFrame:
    """Mean energy per active truck on each date, joined with temperature."""
    d = truck_days.groupby("date")["energy_consumption"].mean().rename("energy")
    return pd.concat([d, temp], axis=1, join="inner")


def pearson_ci(x, y):
    r, p = stats.pearsonr(x, y)
    z, se = np.arctanh(r), 1 / np.sqrt(len(x) - 3)
    return r, p, np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)


def main():
    truck_days = load_truck_days()
    temp = load_daily_temperature()
    fleet = daily_mean(truck_days, temp)

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(15, 6), dpi=180, gridspec_kw={"width_ratios": [3, 2]})

    # Left: fleet-wide scatter, coloured by month, with OLS fit and 95% CI band.
    sc = ax.scatter(fleet["temp"], fleet["energy"], c=fleet.index.month, cmap="viridis", s=22, alpha=0.85)
    fit = stats.linregress(fleet["temp"], fleet["energy"])
    xs = np.linspace(fleet["temp"].min(), fleet["temp"].max(), 100)
    n, xm = len(fleet), fleet["temp"].mean()
    resid_se = np.sqrt(np.sum((fleet["energy"] - (fit.intercept + fit.slope * fleet["temp"])) ** 2) / (n - 2))
    band = 1.96 * resid_se * np.sqrt(1 / n + (xs - xm) ** 2 / np.sum((fleet["temp"] - xm) ** 2))
    yfit = fit.intercept + fit.slope * xs
    ax.fill_between(xs, yfit - band, yfit + band, color="#b8434f", alpha=0.18, lw=0)
    ax.plot(xs, yfit, color="#b8434f", lw=2)
    r, p, lo, hi = pearson_ci(fleet["temp"], fleet["energy"])
    ax.text(0.02, 0.97,
            f"Pearson r = {r:.2f} (95% CI {lo:.2f} to {hi:.2f}), p = {p:.2g}\n"
            f"slope = {fit.slope:.2f} per °C, n = {n} days",
            transform=ax.transAxes, va="top", fontsize=10,
            bbox=dict(boxstyle="round", fc="white", ec="#cccccc"))
    cb = fig.colorbar(sc, ax=ax, pad=0.01)
    cb.set_label("Month")
    ax.set_xlabel("Daily average temperature (°C)")
    ax.set_ylabel("Mean daily energy per active truck (dataset units)")
    ax.set_title("Daily energy vs daily average temperature — all trucks (Jan–Oct 2019)", fontsize=13)
    ax.grid(color="#e5e5e5")
    ax.set_axisbelow(True)

    # Right: correlation per truck category with 95% CI.
    rows = []
    for cat, g in truck_days.groupby("truck_category"):
        d = daily_mean(g, temp)
        rows.append((cat, *pearson_ci(d["temp"], d["energy"])))
    rc = pd.DataFrame(rows, columns=["cat", "r", "p", "lo", "hi"]).sort_values("r")
    y = np.arange(len(rc))
    bx.barh(y, rc["r"], color=[COLORS[c] for c in rc["cat"]], height=0.6)
    bx.errorbar(rc["r"], y, xerr=[rc["r"] - rc["lo"], rc["hi"] - rc["r"]], fmt="none", ecolor="black", capsize=4)
    bx.axvline(0, color="black", lw=0.8)
    bx.set_yticks(y, rc["cat"])
    bx.set_xlim(-1, 1)
    bx.set_xlabel("Pearson r (daily temperature vs daily energy)")
    bx.set_title("Correlation by truck category (95% CI)", fontsize=12)
    bx.grid(axis="x", color="#e5e5e5")
    bx.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(HERE / "daily_energy_vs_temperature.png")
    plt.close(fig)

    print(f"fleet: r={r:.3f} [{lo:.3f}, {hi:.3f}] p={p:.3g} slope={fit.slope:.3f}/°C n={n}")
    print(rc.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
