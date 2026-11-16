"""Sep-Oct daily energy for several trucks: forecast without distance vs known km.

Models (trained Jan-Aug on all trucks, tested Sep-Oct):
  unknown distance + past 7 days : forecast using only information available
      before the day: truck attributes, temperature, month, zone passes, and the
      truck's own previous active days (last day's energy and km, mean and std
      of the last 7 active days, days since last active). No route clusters and
      no same-day distance.
  predicted with known km : uses the truck's actual GPS km that day (upper bound)

Each panel shows one truck: actual energy, the unknown-distance forecast with
its 90% interval, and the known-km prediction. A final panel shows the
distribution of per-truck R^2 across all trucks with enough test days.

Usage: python predict_trucks_panel.py [TruckId ...]
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

from predict_energy import build_dataset
from predict_energy_forecast import HISTORY, STATIC, add_history, fit
from predict_energy_scenarios import run

HERE = Path(__file__).resolve().parent
SPLIT = "2023-09-01"
MIN_TEST_DAYS = 20
DEFAULT_TRUCKS = [13873, 3019, 12263, 1988, 2921, 2705, 8065, 2201]  # one per category, most Sep-Oct days


def per_truck_scores(test: pd.DataFrame, pred: np.ndarray) -> pd.DataFrame:
    t = test.assign(pred=pred)
    rows = []
    for tid, g in t.groupby("TruckId"):
        if len(g) >= MIN_TEST_DAYS and g["energy"].var() > 0:
            rows.append((tid, g["truck_category"].iloc[0], len(g),
                         r2_score(g["energy"], g["pred"]), mean_absolute_error(g["energy"], g["pred"])))
    return pd.DataFrame(rows, columns=["TruckId", "category", "days", "R2", "MAE"])


def main(trucks=None):
    d = add_history(build_dataset())
    train, test = d[d["date"] < SPLIT], d[d["date"] >= SPLIT]
    print("fitting models...")
    forecast = fit(train, test, STATIC + HISTORY, quantiles=True)
    known = run(train, test, 0.0)

    trucks = [t for t in (trucks or DEFAULT_TRUCKS) if t in set(test["TruckId"])]
    scores = per_truck_scores(test, forecast["pred"])
    scores_known = per_truck_scores(test, known["pred"])

    n = len(trucks)
    fig, axes = plt.subplots(n + 1, 1, figsize=(16, 3.1 * (n + 1)), dpi=150)
    for ax, tid in zip(axes, trucks):
        m = (test["TruckId"] == tid).to_numpy()
        tt = test[m]
        y = tt["energy"].to_numpy()
        p, lo, hi, pk = forecast["pred"][m], forecast["lo"][m], forecast["hi"][m], known["pred"][m]
        cov = np.mean((y >= lo) & (y <= hi))
        ax.fill_between(tt["date"], lo, hi, color="#e08a1e", alpha=0.18, lw=0, label="90% interval (unknown distance)")
        ax.plot(tt["date"], y, color="black", lw=1.2, marker="o", ms=2.5, label="Actual")
        ax.plot(tt["date"], p, color="#e08a1e", lw=1.8, label="Forecast: unknown distance + past 7 days")
        ax.plot(tt["date"], pk, color="#2a6ebb", lw=1, ls="--", label="Predicted with known km")
        info = tt.iloc[0]
        attrs = f"duty {info['duty_cycle']}, {info['km']:,.0f} km odometer" if info["in_fleet"] else "not in fleet list"
        ax.set_title(f"Truck {tid} — {info['truck_category']} ({attrs}) | unknown distance + past 7 days: "
                     f"R² {r2_score(y, p):.2f}, MAE {mean_absolute_error(y, p):.1f}, 90% cover {cov:.0%} | "
                     f"known km: R² {r2_score(y, pk):.2f}", fontsize=10)
        ax.set_ylabel("Daily energy")
        ax.set_ylim(0, max(y.max(), hi.max()) * 1.1)
        ax.set_xlim(test["date"].min(), test["date"].max())
        ax.grid(color="#e5e5e5")
        ax.set_axisbelow(True)
    axes[0].legend(frameon=False, loc="upper left", ncol=4, fontsize=9)

    ax = axes[-1]
    bins = np.linspace(-1, 1, 41)
    ax.hist(scores["R2"].clip(-1, 1), bins=bins, color="#e08a1e", alpha=0.75,
            label=f"Unknown distance + past 7 days (median R² {scores['R2'].median():.2f})")
    ax.hist(scores_known["R2"].clip(-1, 1), bins=bins, color="#2a6ebb", alpha=0.45,
            label=f"Predicted with known km (median R² {scores_known['R2'].median():.2f})")
    ax.set_xlabel("Per-truck R² over Sep–Oct (values below −1 shown at −1)")
    ax.set_ylabel("Trucks")
    ax.set_title(f"Per-truck accuracy across all {len(scores)} trucks with ≥{MIN_TEST_DAYS} test days", fontsize=11)
    ax.legend(frameon=False, loc="upper left")
    ax.grid(color="#e5e5e5")
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(HERE / "energy_prediction_trucks_panel.png")
    plt.close(fig)

    y = test["energy"].to_numpy()
    print(f"all truck-days: unknown distance + past 7 days R2 {r2_score(y, forecast['pred']):.3f}, "
          f"MAE {mean_absolute_error(y, forecast['pred']):.2f}, "
          f"90% coverage {np.mean((y >= forecast['lo']) & (y <= forecast['hi'])):.1%}; "
          f"known km R2 {r2_score(y, known['pred']):.3f}")
    print(scores[scores["TruckId"].isin(trucks)].round(2).to_string(index=False))
    print("\nper-truck R2 (unknown distance + past 7 days) by category, median:")
    print(scores.groupby("category")["R2"].agg(["median", "count"]).round(2).to_string())
    print(f"\nshare of trucks with R2 > 0: {np.mean(scores['R2'] > 0):.0%}, > 0.5: {np.mean(scores['R2'] > 0.5):.0%}")


if __name__ == "__main__":
    main([int(a) for a in sys.argv[1:]])
