"""Sep-Oct daily energy for several trucks: forecast without distance vs known km.

Models (trained Jan-Aug on all trucks, tested Sep-Oct):
  unknown distance + past 7 days : forecast using only information available
      before the day: truck attributes, temperature, month, zone passes, and the
      truck's own previous active days (last day's energy and km, mean and std
      of the last 7 active days, days since last active). No route clusters and
      no same-day distance.
  refined clusters + past 7 days : same as above plus planned km from route
      image clusters refined with daily distance (predict_energy_route_cluster_
      refined.py, k=20, weight=0.5, the best setting there); a day is assigned
      to a cluster by its route image only, so actual km is not used.
  predicted with known km : uses the truck's actual GPS km that day (upper bound)

Each panel shows one truck: actual energy, the unknown-distance forecast with
its 90% interval, the refined-cluster forecast and the known-km prediction. A final panel shows the
distribution of per-truck R^2 across all trucks with enough test days.

Usage: python predict_trucks_panel.py [TruckId ...]
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from predict_energy import build_dataset
from predict_energy_forecast import HISTORY, STATIC, add_history, fit
from predict_energy_route_cluster_refined import image_features, planned_km
from predict_energy_scenarios import run

HERE = Path(__file__).resolve().parent
SPLIT = "2023-09-01"
MIN_TEST_DAYS = 20
DEFAULT_TRUCKS = [13873, 3019, 12263, 1988, 2921, 2705, 8065, 2201]  # one per category, most Sep-Oct days
REFINED_K, REFINED_W = 20, 0.5   # best image-only setting in predict_energy_route_cluster_refined.py
REFINED_COLOR = "#7e5fa6"


def per_truck_scores(test: pd.DataFrame, pred: np.ndarray) -> pd.DataFrame:
    t = test.assign(pred=pred)
    rows = []
    for tid, g in t.groupby("TruckId"):
        if len(g) >= MIN_TEST_DAYS and g["energy"].var() > 0:
            rows.append((tid, g["truck_category"].iloc[0], len(g),
                         r2_score(g["energy"], g["pred"]), mean_absolute_error(g["energy"], g["pred"])))
    return pd.DataFrame(rows, columns=["TruckId", "category", "days", "R2", "MAE"])


def mse_table(test: pd.DataFrame, preds: dict, trucks: list) -> list[list[str]]:
    """Rows of the MSE table: subsets, median per-truck MSE, then the panel trucks."""
    names = list(preds)
    t = test.assign(**{f"p{i}": v for i, v in enumerate(preds.values())})
    mse = lambda g: [mean_squared_error(g["energy"], g[f"p{i}"]) for i in range(len(names))]
    img = t["planned_km"].notna()
    rows = [("All truck-days", mse(t), len(t)),
            ("Truck-days with a route image", mse(t[img]), int(img.sum())),
            ("Truck-days without a route image", mse(t[~img]), int((~img).sum()))]
    ok = t.groupby("TruckId").filter(lambda g: len(g) >= MIN_TEST_DAYS and g["energy"].var() > 0)
    per = np.array([mse(g) for _, g in ok.groupby("TruckId")])
    rows.append((f"Median per-truck MSE ({len(per)} trucks)", list(np.median(per, axis=0)), None))
    for tid in trucks:
        g = t[t["TruckId"] == tid]
        rows.append((f"Truck {tid} ({g['truck_category'].iloc[0]})", mse(g), len(g)))
    cells = []
    for label, (u, r, k), n in rows:
        cells.append([label, f"{u:.1f}", f"{r:.1f}", f"{r - u:+.1f} ({(r - u) / u:+.0%})", f"{k:.1f}",
                      "–" if n is None else f"{n:,}"])
    return cells


def main(trucks=None):
    d = add_history(build_dataset())
    # refined-cluster planned km; truck-days without a route image keep NaN (filled with -1 in fit)
    pk = planned_km(d, image_features(d), REFINED_W, REFINED_K)[["TruckId", "date", "planned_km"]]
    d = d.merge(pk, on=["TruckId", "date"], how="left")
    train, test = d[d["date"] < SPLIT], d[d["date"] >= SPLIT]
    print("fitting models...")
    forecast = fit(train, test, STATIC + HISTORY, quantiles=True)
    refined = fit(train, test, STATIC + ["planned_km"] + HISTORY)
    known = run(train, test, 0.0)

    trucks = [t for t in (trucks or DEFAULT_TRUCKS) if t in set(test["TruckId"])]
    scores = per_truck_scores(test, forecast["pred"])
    scores_refined = per_truck_scores(test, refined["pred"])
    scores_known = per_truck_scores(test, known["pred"])

    n = len(trucks)
    fig, axes = plt.subplots(n + 2, 1, figsize=(16, 3.1 * (n + 1) + 0.32 * (n + 5)), dpi=150,
                             gridspec_kw={"height_ratios": [3.1] * (n + 1) + [0.32 * (n + 5)]})
    for ax, tid in zip(axes, trucks):
        m = (test["TruckId"] == tid).to_numpy()
        tt = test[m]
        y = tt["energy"].to_numpy()
        p, lo, hi, pk = forecast["pred"][m], forecast["lo"][m], forecast["hi"][m], known["pred"][m]
        pr = refined["pred"][m]
        cov = np.mean((y >= lo) & (y <= hi))
        ax.fill_between(tt["date"], lo, hi, color="#e08a1e", alpha=0.18, lw=0, label="90% interval (unknown distance)")
        ax.plot(tt["date"], y, color="black", lw=1.2, marker="o", ms=2.5, label="Actual")
        ax.plot(tt["date"], p, color="#e08a1e", lw=1.8, label="Forecast: unknown distance + past 7 days")
        ax.plot(tt["date"], pr, color=REFINED_COLOR, lw=1.4, label="Forecast: refined clusters + past 7 days")
        ax.plot(tt["date"], pk, color="#2a6ebb", lw=1, ls="--", label="Predicted with known km")
        info = tt.iloc[0]
        attrs = f"duty {info['duty_cycle']}, {info['km']:,.0f} km odometer" if info["in_fleet"] else "not in fleet list"
        ax.set_title(f"Truck {tid} — {info['truck_category']} ({attrs}) | unknown distance + past 7 days: "
                     f"R² {r2_score(y, p):.2f}, MAE {mean_absolute_error(y, p):.1f}, 90% cover {cov:.0%} | "
                     f"refined clusters: R² {r2_score(y, pr):.2f} | known km: R² {r2_score(y, pk):.2f}", fontsize=10)
        ax.set_ylabel("Daily energy")
        ax.set_ylim(0, max(y.max(), hi.max()) * 1.1)
        ax.set_xlim(test["date"].min(), test["date"].max())
        ax.grid(color="#e5e5e5")
        ax.set_axisbelow(True)
    axes[0].legend(frameon=False, loc="upper left", ncol=5, fontsize=9)

    ax = axes[-2]
    bins = np.linspace(-1, 1, 41)
    ax.hist(scores["R2"].clip(-1, 1), bins=bins, color="#e08a1e", alpha=0.75,
            label=f"Unknown distance + past 7 days (median R² {scores['R2'].median():.2f})")
    ax.hist(scores_refined["R2"].clip(-1, 1), bins=bins, color=REFINED_COLOR, alpha=0.5,
            label=f"Refined clusters + past 7 days (median R² {scores_refined['R2'].median():.2f})")
    ax.hist(scores_known["R2"].clip(-1, 1), bins=bins, color="#2a6ebb", alpha=0.45,
            label=f"Predicted with known km (median R² {scores_known['R2'].median():.2f})")
    ax.set_xlabel("Per-truck R² over Sep–Oct (values below −1 shown at −1)")
    ax.set_ylabel("Trucks")
    ax.set_title(f"Per-truck accuracy across all {len(scores)} trucks with ≥{MIN_TEST_DAYS} test days", fontsize=11)
    ax.legend(frameon=False, loc="upper left")
    ax.grid(color="#e5e5e5")
    ax.set_axisbelow(True)

    ax = axes[-1]
    ax.axis("off")
    cells = mse_table(test, {"u": forecast["pred"], "r": refined["pred"], "k": known["pred"]}, trucks)
    cols = ["Subset", "Unknown distance\n+ past 7 days", "Refined clusters\n+ past 7 days",
            "Change\n(refined vs unknown)", "Predicted with\nknown km", "Days"]
    tab = ax.table(cellText=cells, colLabels=cols, loc="center", cellLoc="center",
                   colWidths=[0.27, 0.14, 0.14, 0.16, 0.13, 0.08], bbox=[0, 0, 1, 1])
    tab.auto_set_font_size(False)
    tab.set_fontsize(10)
    for (r, c), cell in tab.get_celld().items():
        cell.set_edgecolor("#d0d0d0")
        if r == 0:
            cell.set_height(cell.get_height() * 2)   # two-line headers; bbox rescales the rest
            cell.set_facecolor("#f0f0f0")
            cell.set_text_props(weight="bold")
        elif c in (1, 2):   # bold the better of the two forecasts (neither uses actual km)
            u, rr = float(cells[r - 1][1]), float(cells[r - 1][2])
            if (c == 1) == (u < rr):
                cell.set_text_props(weight="bold")
        if c == 0 and r > 0:
            cell.set_text_props(ha="left")
            cell.PAD = 0.02
    ax.set_title("Daily energy MSE, Sep–Oct (lower is better; bold = better of the two forecasts without actual km)",
                 fontsize=11)

    fig.tight_layout()
    fig.savefig(HERE / "energy_prediction_trucks_panel.png")
    plt.close(fig)

    y = test["energy"].to_numpy()
    print(f"all truck-days: unknown distance + past 7 days R2 {r2_score(y, forecast['pred']):.3f}, "
          f"MAE {mean_absolute_error(y, forecast['pred']):.2f}, "
          f"90% coverage {np.mean((y >= forecast['lo']) & (y <= forecast['hi'])):.1%}; "
          f"refined clusters + past 7 days R2 {r2_score(y, refined['pred']):.3f}, "
          f"MAE {mean_absolute_error(y, refined['pred']):.2f}; "
          f"known km R2 {r2_score(y, known['pred']):.3f}")
    print(scores[scores["TruckId"].isin(trucks)].round(2).to_string(index=False))
    print("\nper-truck R2 (unknown distance + past 7 days) by category, median:")
    print(scores.groupby("category")["R2"].agg(["median", "count"]).round(2).to_string())
    print("\nper-truck R2 (refined clusters + past 7 days) by category, median:")
    print(scores_refined.groupby("category")["R2"].agg(["median", "count"]).round(2).to_string())
    print(f"\nshare of trucks with R2 > 0: {np.mean(scores['R2'] > 0):.0%}, > 0.5: {np.mean(scores['R2'] > 0.5):.0%}")


if __name__ == "__main__":
    main([int(a) for a in sys.argv[1:]])
