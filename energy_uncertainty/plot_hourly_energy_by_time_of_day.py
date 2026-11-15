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
DUTY_CYCLE = HERE.parent / "Other_data" / "West_Ang_Fleet_duty_cycle.csv"
DUTY_COLORS = {"LIGHT": "#3a9a6a", "MEDIUM": "#2a6ebb", "HEAVY": "#b8434f"}
FLEET = HERE.parent / "Other_data" / "West_Ang_Fleet.csv"
AGE_COLORS = ["#3a9a6a", "#2a6ebb", "#b8434f"]  # young -> old

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

    cats = sorted(hourly["truck_category"].unique())
    plot_groups(hourly.assign(group=hourly["truck_category"]), cats, [COLORS[c] for c in cats],
                "truck category", HERE / "truck_category_average_energy_by_time_of_day_ci.png")

    plot_duty_cycle(hourly)
    plot_age_groups(hourly)


def plot_duty_cycle(hourly: pd.DataFrame) -> None:
    """Energy by fleet duty cycle (trucks without a duty cycle are dropped)."""
    duty = pd.read_csv(DUTY_CYCLE).rename(columns={"VehicleId": "TruckId", "Duty Cycle": "group"})
    h = hourly.merge(duty, on="TruckId", how="inner")
    order = [d for d in DUTY_COLORS if d in set(h["group"])]
    plot_groups(h, order, [DUTY_COLORS[d] for d in order], "duty cycle",
                HERE / "duty_cycle_average_energy_by_time_of_day_ci.png")


def plot_age_groups(hourly: pd.DataFrame) -> None:
    """Energy by vehicle age, split into three equal-count groups (tertiles)."""
    fleet = pd.read_csv(FLEET, usecols=["VehicleId", "Age"]).dropna()
    fleet = fleet.rename(columns={"VehicleId": "TruckId"})
    fleet["TruckId"] = fleet["TruckId"].astype(int)
    fleet = fleet[fleet["TruckId"].isin(hourly["TruckId"].unique())]
    edges = fleet["Age"].quantile([0, 1 / 3, 2 / 3, 1]).to_numpy()
    labels = [f"{edges[0]:.1f}–{edges[1]:.1f} yr", f"{edges[1]:.1f}–{edges[2]:.1f} yr", f"{edges[2]:.1f}–{edges[3]:.1f} yr"]
    fleet["group"] = pd.qcut(fleet["Age"], 3, labels=labels)
    fleet["group"] = fleet["group"].astype(str)
    h = hourly.merge(fleet[["TruckId", "group"]], on="TruckId", how="inner")
    plot_groups(h, labels, AGE_COLORS, "vehicle age",
                HERE / "age_group_average_energy_by_time_of_day_ci.png")


def plot_groups(h: pd.DataFrame, order: list[str], colors: list[str], what: str, out: Path) -> None:
    """Hourly profile (left) and mean daily energy (right) for each value of ``h['group']``."""
    n_trucks = h.groupby("group")["TruckId"].nunique()

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(15, 6), dpi=180, gridspec_kw={"width_ratios": [3, 1]})

    s = summarize(h, ["group", "hour_of_day"])
    for g, c in zip(order, colors):
        d = s[s["group"] == g]
        ax.fill_between(d["hour_of_day"], d["lo"], d["hi"], color=c, alpha=0.18, lw=0)
        ax.plot(d["hour_of_day"], d["mean"], color=c, lw=2, marker="o", ms=4, label=f"{g} (n={n_trucks[g]} trucks)")
    style(ax, f"Average hourly energy by time of day — by {what} (shaded: 95% CI)")
    many = len(order) > 4
    if many:  # leave room above the lines for the two-column legend
        ax.set_ylim(0, s["hi"].max() * 1.35)
    ax.legend(frameon=False, loc="upper left", ncol=2 if many else 1, fontsize=9 if many else 10)

    daily = h.groupby(["TruckId", "group", "month", "day"], as_index=False)["energy_consumption"].sum()
    ds = summarize(daily, ["group"]).set_index("group").loc[order]
    x = np.arange(len(order))
    bx.bar(x, ds["mean"], color=colors, width=0.6)
    bx.errorbar(x, ds["mean"], yerr=[ds["mean"] - ds["lo"], ds["hi"] - ds["mean"]], fmt="none", ecolor="black", capsize=5)
    for xi, hi, m in zip(x, ds["hi"], ds["mean"]):
        bx.text(xi, hi + ds["hi"].max() * 0.02, f"{m:.1f}", ha="center", va="bottom", fontsize=8 if many else 10)
    bx.set_ylim(0, ds["hi"].max() * 1.12)
    bx.set_xticks(x, order, rotation=45 if many else 0, ha="right" if many else "center")
    bx.set_title("Mean daily energy per active truck-day", fontsize=12)
    bx.set_ylabel("Energy per day (dataset units)")
    bx.grid(axis="y", color="#e5e5e5")
    bx.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


if __name__ == "__main__":
    main()
