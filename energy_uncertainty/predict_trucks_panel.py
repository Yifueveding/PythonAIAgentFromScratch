"""Sep-Oct daily energy forecasts for several trucks, one per category.

Models (trained Jan-Aug on all trucks, as in predict_energy_route_cluster_refined.py):
  refined cluster km : planned km from distance-refined route clusters, assigned
                       by route image only (realistic forecast)
  predicted with known km : uses the truck's actual GPS km that day (upper bound)

Each panel shows one truck: actual energy, the refined-cluster forecast with its
90% interval, and the actual-km prediction for reference. A final panel shows
the distribution of per-truck R^2 across all trucks with enough test days.

Usage: python predict_trucks_panel.py [TruckId ...]   (default: one per category)
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

from predict_energy import build_dataset
from predict_energy_route_cluster import fit_with_planned
from predict_energy_route_cluster_refined import SPLIT, image_features, planned_km
from predict_energy_scenarios import run

HERE = Path(__file__).resolve().parent
BEST_K, BEST_W = 20, 0.5  # best image-only setting from predict_energy_route_cluster_refined.py
MIN_TEST_DAYS = 20


def per_truck_scores(test: pd.DataFrame, pred: np.ndarray) -> pd.DataFrame:
    t = test.assign(pred=pred)
    rows = []
    for tid, g in t.groupby("TruckId"):
        if len(g) >= MIN_TEST_DAYS and g["energy"].var() > 0:
            rows.append((tid, g["truck_category"].iloc[0], len(g),
                         r2_score(g["energy"], g["pred"]), mean_absolute_error(g["energy"], g["pred"])))
    return pd.DataFrame(rows, columns=["TruckId", "category", "days", "R2", "MAE"])


def main(trucks=None):
    d = build_dataset()
    d = planned_km(d, image_features(d), BEST_W, BEST_K)
    train, test = d[d["date"] < SPLIT], d[d["date"] >= SPLIT]
    print("fitting models...")
    refined = fit_with_planned(train, test)
    actual = run(train, test, 0.0)

    if not trucks:
        counts = test.groupby(["truck_category", "TruckId"]).size().rename("n").reset_index()
        trucks = counts.sort_values("n", ascending=False).groupby("truck_category").head(1)
        trucks = trucks.sort_values("truck_category")["TruckId"].tolist()
    trucks = [t for t in trucks if t in set(test["TruckId"])]

    scores = per_truck_scores(test, refined["pred"])
    scores_actual = per_truck_scores(test, actual["pred"])

    n = len(trucks)
    fig, axes = plt.subplots(n + 1, 1, figsize=(16, 3.1 * (n + 1)), dpi=150)
    for ax, tid in zip(axes, trucks):
        m = (test["TruckId"] == tid).to_numpy()
        tt = test[m]
        y = tt["energy"].to_numpy()
        p, lo, hi, pa = refined["pred"][m], refined["lo"][m], refined["hi"][m], actual["pred"][m]
        cov = np.mean((y >= lo) & (y <= hi))
        ax.fill_between(tt["date"], lo, hi, color="#7e5fa6", alpha=0.18, lw=0, label="90% interval (route-cluster)")
        ax.plot(tt["date"], y, color="black", lw=1.2, marker="o", ms=2.5, label="Actual")
        ax.plot(tt["date"], p, color="#7e5fa6", lw=1.8, label="Forecast: refined route-cluster km")
        ax.plot(tt["date"], pa, color="#2a6ebb", lw=1, ls="--", label="Predicted with known km")
        info = tt.iloc[0]
        attrs = f"duty {info['duty_cycle']}, {info['km']:,.0f} km odometer" if info["in_fleet"] else "not in fleet list"
        ax.set_title(f"Truck {tid} — {info['truck_category']} ({attrs}) | route-cluster forecast: "
                     f"R² {r2_score(y, p):.2f}, MAE {mean_absolute_error(y, p):.1f}, 90% cover {cov:.0%} | "
                     f"predicted with known km: R² {r2_score(y, pa):.2f}", fontsize=10)
        ax.set_ylabel("Daily energy")
        ax.set_ylim(0, max(y.max(), hi.max()) * 1.1)
        ax.set_xlim(test["date"].min(), test["date"].max())
        ax.grid(color="#e5e5e5")
        ax.set_axisbelow(True)
    axes[0].legend(frameon=False, loc="upper left", ncol=4, fontsize=9)

    ax = axes[-1]
    bins = np.linspace(-1, 1, 41)
    ax.hist(scores["R2"].clip(-1, 1), bins=bins, color="#7e5fa6", alpha=0.75,
            label=f"Refined route-cluster km (median R² {scores['R2'].median():.2f})")
    ax.hist(scores_actual["R2"].clip(-1, 1), bins=bins, color="#2a6ebb", alpha=0.45,
            label=f"Predicted with known km (median R² {scores_actual['R2'].median():.2f})")
    ax.set_xlabel("Per-truck R² over Sep–Oct (values below −1 shown at −1)")
    ax.set_ylabel("Trucks")
    ax.set_title(f"Per-truck accuracy across all {len(scores)} trucks with ≥{MIN_TEST_DAYS} test days", fontsize=11)
    ax.legend(frameon=False, loc="upper left")
    ax.grid(color="#e5e5e5")
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(HERE / "energy_prediction_trucks_panel.png")
    plt.close(fig)

    print(scores[scores["TruckId"].isin(trucks)].round(2).to_string(index=False))
    print("\nper-truck R2 (refined cluster km) by category, median:")
    print(scores.groupby("category")["R2"].agg(["median", "count"]).round(2).to_string())
    print(f"\nshare of trucks with R2 > 0: {np.mean(scores['R2'] > 0):.0%}, > 0.5: {np.mean(scores['R2'] > 0.5):.0%}")


if __name__ == "__main__":
    main([int(a) for a in sys.argv[1:]])
